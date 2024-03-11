### imports
import os
import shutil
import pandas as pd
import numpy as np
import json

from SimpleITK import (
    DICOMOrientImageFilter,
    sitkLinear,
    sitkNearestNeighbor,
    ReadImage,
    Resample,
    ResampleImageFilter,
    WriteImage,
    GetArrayFromImage,
    GetImageFromArray,
    N4BiasFieldCorrectionImageFilter,
    RescaleIntensity,
    sitkFloat32,
    sitkUInt8,
    Cast,
)

from pathlib import Path
from subprocess import run
from tqdm import tqdm
from preprocessing.utils import (
    source_external_software,
    check_required_columns,
)
from preprocessing.synthmorph import synthmorph_registration
from typing import Sequence, Literal
from scipy.ndimage import (
    binary_fill_holes,
    binary_closing,
    distance_transform_edt,
    generate_binary_structure,
)
from cc3d import connected_components
from concurrent.futures import ThreadPoolExecutor, as_completed


def copy_metadata(row: dict, preprocessing_args: dict) -> None:
    """
    Copy the metadata file paired with the original nifti file (and optionally the
    corresponding segmentation) and add the preprocessing arguments into a new metafile
    to be paired with the preprocessing outputs.

    Parameters
    __________
    row: dict
        A row of a DataFrame represented as a dictionary. It is expected to have a 'nifti'
        key and optionally 'seg'.
    preprocessing_args: dict
        A dictionary containing the arguments originally provided to 'preprocess_study' or
        'preprocess_from_csv'.

    Returns
    _______
    None
        A metadata json is saved out to be paired with the preprocessed outputs.

    """
    original_metafile = row["nifti"].replace(".nii.gz", ".json")
    if Path(original_metafile).exists():
        try:
            with open(original_metafile, "r") as json_file:
                data = json.load(json_file)
        except Exception:
            data = original_metafile
        meta_dict = {
            "source_file": row["nifti"],
            "original_metafile": data,
            "preprocessing_args": preprocessing_args,
        }
        preprocessed_metafile = row[preprocessing_args["pipeline_key"]].replace(
            ".nii.gz", ".json"
        )
        with open(preprocessed_metafile, "w") as json_file:
            json.dump(
                meta_dict, json_file, sort_keys=True, indent=2, separators=(",", ": ")
            )
    else:
        meta_dict = {
            "source_file": row["nifti"],
            "original_metafile": None,
            "preprocessing_args": preprocessing_args,
        }
        preprocessed_metafile = row[preprocessing_args["pipeline_key"]].replace(
            ".nii.gz", ".json"
        )
        with open(preprocessed_metafile, "w") as json_file:
            json.dump(
                meta_dict, json_file, sort_keys=True, indent=2, separators=(",", ": ")
            )

    if "seg" in row and not pd.isna(row["seg"]):
        original_metafile = row["seg"].replace(".nii.gz", ".json")
        if Path(original_metafile).exists():
            try:
                with open(original_metafile, "r") as json_file:
                    data = json.load(json_file)
            except Exception:
                data = original_metafile
            meta_dict = {
                "source_file": row["nifti"],
                "original_metafile": data,
                "preprocessing_args": preprocessing_args,
            }
            preprocessed_metafile = row[
                f"{preprocessing_args['pipeline_key']}_seg"
            ].replace(".nii.gz", ".json")
            with open(preprocessed_metafile, "w") as json_file:
                json.dump(
                    meta_dict,
                    json_file,
                    sort_keys=True,
                    indent=2,
                    separators=(",", ": "),
                )
        else:
            meta_dict = {
                "source_file": row["nifti"],
                "original_metafile": None,
                "preprocessing_args": preprocessing_args,
            }
            preprocessed_metafile = row[
                f"{preprocessing_args['pipeline_key']}_seg"
            ].replace(".nii.gz", ".json")
            with open(preprocessed_metafile, "w") as json_file:
                json.dump(
                    meta_dict,
                    json_file,
                    sort_keys=True,
                    indent=2,
                    separators=(",", ": "),
                )


def verify_reg(
    fixed_image_path, moving_image_path, interp_method="linear", verbose=False
):
    def get_affine(image):
        affine = np.eye(4)
        affine[:3, :3] = np.reshape(image.GetDirection(), (3, 3))
        affine[:3, 3] = image.GetOrigin()
        return affine

    if verbose:
        print(f"{moving_image_path} is being checked against {fixed_image_path}")

    fixed_image = ReadImage(fixed_image_path)
    moving_image = ReadImage(moving_image_path)
    same_shape = fixed_image.GetSize() == moving_image.GetSize()

    if verbose:
        print(
            f"moving: {moving_image.GetSize()}, fixed; {fixed_image.GetSize()}, {same_shape}"
        )

    fixed_affine = get_affine(fixed_image)
    moving_affine = get_affine(moving_image)
    close_affines = np.allclose(fixed_affine, moving_affine, atol=1e-3)

    if not same_shape or not close_affines:
        if verbose:
            if not same_shape:
                print(
                    f"shapes do not match: {fixed_image.GetSize()} {moving_image.GetSize()}"
                )
            if not close_affines:
                print(f"affines:\n{fixed_affine} \n{moving_affine}")
        resampler = ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        if interp_method == "linear":
            resampler.SetInterpolator(sitkLinear)
        elif interp_method == "nearest":
            resampler.SetInterpolator(sitkNearestNeighbor)

        resampled_image = resampler.Execute(moving_image)

        WriteImage(resampled_image, moving_image_path)

        return False

    else:
        return True


def local_reg(
    row, pipeline_key, fixed_image_path, model="affine", verbose=False, debug=False
):
    preprocessed_file = row[pipeline_key]

    if debug:
        output_file = preprocessed_file.replace(".nii.gz", "_localreg.nii.gz")
        row[pipeline_key] = output_file

    else:
        output_file = preprocessed_file

    moving_image_path = preprocessed_file.replace(".nii.gz", "_SS.nii.gz")

    accompanying_images = [{"moving": preprocessed_file, "out_moving": output_file}]

    if "seg" in row and not pd.isna(row["seg"]):
        preprocessed_seg = row[f"{pipeline_key}_seg"]
        if debug:
            output_seg = preprocessed_seg.replace(".nii.gz", "_localreg.nii.gz")
            row[f"{pipeline_key}_seg"] = output_seg

        else:
            output_seg = preprocessed_seg

        accompanying_images.append(
            {
                "moving": preprocessed_seg,
                "out_moving": output_seg,
                "interp_method": "nearest",
            }
        )

    synthmorph_registration(
        moving_image_path,
        fixed_image_path,
        accompanying_images=accompanying_images,
        m=model,
    )
    if verbose:
        print(
            f"Registered files generated to {[d['out_moving'] for d in accompanying_images]}"
        )

    redo_registrations = not verify_reg(fixed_image_path, output_file, verbose=verbose)

    if redo_registrations:
        if verbose:
            print(
                "Registered images do not share the same affine or shape and require resampling."
            )

        if len(accompanying_images) > 1:
            for accompanying_image in accompanying_images:
                moving_image_path = accompanying_image["out_moving"]
                interp_method = accompanying_image.get("interp_method", "linear")

                verify_reg(
                    fixed_image_path,
                    moving_image_path,
                    interp_method=interp_method,
                    verbose=verbose,
                )

            if verbose:
                print("Resampling completed.")

    return row


def long_reg(
    rows,
    pipeline_key,
    fixed_image_path,
    study_SS_mask_file,
    model="affine",
    verbose=False,
    debug=False,
):
    moving_image_path = rows[0][pipeline_key].replace(".nii.gz", "_SS.nii.gz")
    if debug:
        moved_image_path = moving_image_path.replace(".nii.gz", "_longreg.nii.gz")

    else:
        moved_image_path = moving_image_path

    accompanying_images = [
        {
            "moving": study_SS_mask_file,
            "out_moving": study_SS_mask_file.replace(".nii.gz", "_longreg.nii.gz"),
            "interp_method": "nearest",
        }
    ]

    for i, row in enumerate(rows):
        preprocessed_file = row[pipeline_key]

        if debug:
            output_file = preprocessed_file.replace(".nii.gz", "_longreg.nii.gz")
            row[pipeline_key] = output_file

        else:
            output_file = preprocessed_file

        accompanying_images.append(
            {"moving": preprocessed_file, "out_moving": output_file}
        )

        if "seg" in row and not pd.isna(row["seg"]):
            preprocessed_seg = row[f"{pipeline_key}_seg"]
            if debug:
                output_seg = preprocessed_seg.replace(".nii.gz", "_longreg.nii.gz")
                row[f"{pipeline_key}_seg"] = output_seg

            else:
                output_seg = preprocessed_seg

            accompanying_images.append(
                {
                    "moving": preprocessed_seg,
                    "out_moving": output_seg,
                    "interp_method": "nearest",
                }
            )
        rows[i] = row

    synthmorph_registration(
        moving_image_path,
        fixed_image_path,
        o=moved_image_path,
        accompanying_images=accompanying_images,
        m=model,
    )
    if verbose:
        print(
            f"Registered files generated to {[d['out_moving'] for d in accompanying_images]}"
        )

    redo_registrations = not verify_reg(
        fixed_image_path, moved_image_path, verbose=verbose
    )

    if redo_registrations:
        if verbose:
            print(
                "Registered images do not share the same affine or shape and require resampling."
            )

        if len(accompanying_images) > 1:
            for accompanying_image in accompanying_images:
                moving_image_path = accompanying_image["out_moving"]
                interp_method = accompanying_image.get("interp_method", "linear")

                verify_reg(
                    fixed_image_path,
                    moving_image_path,
                    interp_method=interp_method,
                    verbose=verbose,
                )

            if verbose:
                print("Resampling completed.")

    return rows


def fill_foreground_mask(initial_foreground: np.ndarray):
    foreground_cc = connected_components(initial_foreground)
    ccs, counts = np.unique(foreground_cc, return_counts=True)

    sorted_ccs = sorted(
        [(cc, count) if cc != 0 else (0, 0) for cc, count in zip(ccs, counts)],
        key=lambda x: x[1],
    )

    largest_cc = sorted_ccs[-1][0]

    foreground = (foreground_cc == largest_cc).astype(int)

    struct_2d = generate_binary_structure(2, 2)

    for z in range(foreground.shape[0]):
        foreground_slice = foreground[z, ...]
        if 1 not in np.unique(foreground_slice):
            continue
        border_mask = np.ones_like(foreground_slice)
        border_mask[0, :] = border_mask[-1, :] = border_mask[:, 0] = border_mask[
            :, -1
        ] = 0
        distance = distance_transform_edt(border_mask)

        iterations = int(distance[foreground_slice == 1].min())

        foreground[z, ...] = binary_fill_holes(
            binary_closing(
                foreground_slice, structure=struct_2d, iterations=iterations
            ),
            structure=struct_2d,
        )

    return foreground.astype(int)


def preprocess_study(
    study_df: pd.DataFrame,
    preprocessed_dir: Path | str,
    pipeline_key: str,
    registration_key: str = "T1Post",
    registration_target: str | None = None,
    registration_model: Literal["rigid", "affine", "joint", "deform"] = "affine",
    orientation: str = "RAS",
    spacing: str = "1,1,1",
    skullstrip: bool = True,
    pre_skullstripped: bool = False,
    binarize_seg: bool = False,
    verbose: bool = False,
    source_software: bool = True,
    check_columns: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Preprocess a single study from a DataFrame.

    Parameters
    __________
    study_df: pd.DataFrame
        A DataFrame containing nifti location and information required for the output file names
        for a single study. It must contain the columns: 'nifti', 'Anon_PatientID', 'Anon_StudyID',
        'StudyInstanceUID', 'SeriesInstanceUID', 'NormalizedSeriesDescription', and 'SeriesType'.
    preprocessed_dir: Path
        The directory that will contain the preprocessed NIfTI files.
    pipeline_key: str
        The key that will be added to the DataFrame to indicate the new locations of preprocessed files.
        Defaults to 'preprocessed'.
    registration_key: str
        The value that will be used to select the fixed image during registration. This should correspond
        to a value within the 'NormalizedSeriesDescription' column in the csv. If you have segmentation
        files in your data. They should correspond to this same series. Defaults to 'T1Post'.
    registration_target: str | None
        The location of the file that will be used as the fixed image for the purposes of registration.
    orientation: str
        The orientation standard that you wish to set for preprocessed data. Defaults to 'RAI'."
    spacing: str
        A comma delimited list indicating the desired spacing of preprocessed data. Measurements
        are in mm. Defaults to '1,1,1'.
    skullstrip: bool
        Whether to apply skullstripping to preprocessed data. Skullstripping will be applied by default.
    pre_skullstripped: bool
        Whether the input data is already skullstripped. Skullstripping will not be applied if specified.
    binarize_seg: bool
        Whether to binarize segmentations. Not recommended for multi-class labels. Binarization is not
        applied by default.
    verbose: bool
        Whether to print additional information related like commands and their arguments are printed.
    source_software: bool
        Whether to call `source_external_software` to add software required for preprocessing. Defaults
        to True.
    check_columns: bool
        Whether to check `study_df` for required columns. Defaults to True.
    debug: bool
        Whether to run in 'debug mode' where each step is saved with an individual name and intermediate
        files are not deleted. Dafaults to False.

    Returns
    _______
    pd.DataFrame:
        A Dataframe with added column f'{pipeline_key}' and optionally f'{pipeline_key}_seg' to indicate
        the locations of the preprocessing outputs.

    """
    if source_software:
        source_external_software()

    if check_columns:
        required_columns = [
            "nifti",
            "Anon_PatientID",
            "Anon_StudyID",
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "NormalizedSeriesDescription",
            "SeriesType",
        ]
        optional_columns = ["seg"]

        check_required_columns(study_df, required_columns, optional_columns)

    preprocessed_dir = Path(preprocessed_dir)

    filtered_df = (
        study_df.copy()
        .dropna(subset="NormalizedSeriesDescription")
        .sort_values(
            ["NormalizedSeriesDescription"],
            key=lambda x: (x != registration_key).astype(int),
        )
    )
    if filtered_df.empty:
        return study_df

    anon_patientID = filtered_df.loc[filtered_df.index[0], "Anon_PatientID"]
    anon_studyID = filtered_df.loc[filtered_df.index[0], "Anon_StudyID"]

    rows = filtered_df.to_dict("records")
    n = len(rows)

    sitk_im_cache = {}

    # must enforce one normalizedseries description per study
    ### copy files to new location
    for i in range(n):
        output_dir = (
            preprocessed_dir / anon_patientID / anon_studyID / rows[i]["SeriesType"]
        )
        os.makedirs(output_dir, exist_ok=True)

        input_file = rows[i]["nifti"]
        preprocessed_file = output_dir / os.path.basename(input_file)
        shutil.copy(input_file, preprocessed_file)

        if os.path.exists(preprocessed_file):
            rows[i][pipeline_key] = str(preprocessed_file)

        else:
            os.makedirs(output_dir, exist_ok=True)

            try:
                shutil.copy(input_file, preprocessed_file)

            except Exception:
                error = f"Could not create {preprocessed_file}"
                print(error)
                e = open(f"{preprocessed_dir}/errors.txt", "a")
                e.write(f"{error}\n")
                setattr(study_df, "failed_preprocessing", True)
                return study_df

        if "seg" in rows[i] and not pd.isna(rows[i]["seg"]):
            input_file = rows[i]["seg"]
            preprocessed_seg = output_dir / os.path.basename(input_file)
            shutil.copy(input_file, preprocessed_seg)

            if os.path.exists(preprocessed_seg):
                rows[i][f"{pipeline_key}_seg"] = str(preprocessed_seg)

            else:
                os.makedirs(output_dir, exist_ok=True)

                try:
                    shutil.copy(input_file, preprocessed_seg)

                except Exception:
                    error = f"Could not create {preprocessed_seg}"
                    print(error)
                    e = open(f"{preprocessed_dir}/errors.txt", "a")
                    e.write(f"{error}\n")
                    setattr(study_df, "failed_preprocessing", True)
                    return study_df

    ### Optionally enforce binary segmentations
    if binarize_seg:
        for i in range(n):
            if "seg" in rows[i] and not pd.isna(rows[i]["seg"]):
                preprocessed_seg = rows[i][f"{pipeline_key}_seg"]

                if debug:
                    output_seg = preprocessed_seg.replace(".nii.gz", "_binary.nii.gz")
                    rows[i][f"{pipeline_key}_seg"] = output_seg

                else:
                    output_seg = preprocessed_seg

                nifti = ReadImage(preprocessed_seg)
                array = GetArrayFromImage(nifti)

                array = (array >= 1).astype(int)

                output_nifti = GetImageFromArray(array)
                output_nifti.CopyInformation(nifti)

                sitk_im_cache[output_seg] = output_nifti

                if debug:
                    WriteImage(output_nifti, output_seg)

    ### RAS
    orienter = DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation("RAS")

    for i in range(n):
        preprocessed_file = rows[i][pipeline_key]

        if debug:
            output_file = preprocessed_file.replace(".nii.gz", "_RAS.nii.gz")
            rows[i][pipeline_key] = output_file

        else:
            output_file = preprocessed_file

        nifti = ReadImage(preprocessed_file)
        output_nifti = orienter.Execute(nifti)

        sitk_im_cache[output_file] = output_nifti

        if debug:
            WriteImage(output_nifti, output_file)

        if "seg" in rows[i] and not pd.isna(rows[i]["seg"]):
            preprocessed_seg = rows[i][f"{pipeline_key}_seg"]

            if debug:
                output_seg = preprocessed_seg.replace(".nii.gz", "_RAS.nii.gz")
                rows[i][f"{pipeline_key}_seg"] = output_seg

            else:
                output_seg = preprocessed_seg

            nifti = sitk_im_cache.get(preprocessed_seg, ReadImage(preprocessed_seg))
            output_nifti = orienter.Execute(nifti)

            sitk_im_cache[output_seg] = output_nifti

            if debug:
                WriteImage(output_nifti, output_seg)

    ### Spacing
    new_spacing = [float(s) for s in spacing.split(",")]
    for i in range(n):
        preprocessed_file = rows[i][pipeline_key]

        if debug:
            output_file = preprocessed_file.replace(".nii.gz", "_spacing.nii.gz")
            rows[i][pipeline_key] = output_file

        else:
            output_file = preprocessed_file

        nifti = sitk_im_cache[preprocessed_file]
        original_spacing = nifti.GetSpacing()
        original_size = nifti.GetSize()
        new_size = [
            int(round(osz * osp / ns))
            for osz, osp, ns in zip(original_size, original_spacing, new_spacing)
        ]
        output_nifti = Resample(
            nifti,
            new_size,
            interpolator=sitkLinear,
            outputOrigin=nifti.GetOrigin(),
            outputSpacing=new_spacing,
            outputDirection=nifti.GetDirection(),
        )

        sitk_im_cache[output_file] = output_nifti
        WriteImage(output_nifti, output_file)

        if "seg" in rows[i] and not pd.isna(rows[i]["seg"]):
            preprocessed_seg = rows[i][f"{pipeline_key}_seg"]

            if debug:
                output_seg = preprocessed_seg.replace(".nii.gz", "_spacing.nii.gz")
                rows[i][f"{pipeline_key}_seg"] = output_seg

            else:
                output_seg = preprocessed_seg

            nifti = sitk_im_cache[preprocessed_seg]
            original_spacing = nifti.GetSpacing()
            original_size = nifti.GetSize()
            new_size = [
                int(round(osz * osp / ns))
                for osz, osp, ns in zip(original_size, original_spacing, new_spacing)
            ]
            output_nifti = Resample(
                nifti,
                new_size,
                interpolator=sitkNearestNeighbor,
                outputOrigin=nifti.GetOrigin(),
                outputSpacing=new_spacing,
                outputDirection=nifti.GetDirection(),
            )

            WriteImage(output_nifti, output_seg)

    ### Loose Skullstrip
    if pre_skullstripped:
        for i in range(n):
            preprocessed_file = rows[i][pipeline_key]
            SS_file = preprocessed_file.replace(".nii.gz", "_SS.nii.gz")
            SS_mask = preprocessed_file.replace(".nii.gz", "_SS_mask.nii.gz")

            nifti = sitk_im_cache[preprocessed_file]
            
            WriteImage(nifti, SS_file)

            array = GetArrayFromImage(nifti)
            ss_array = 1 - (array == 0).astype(int)

            out_nifti = GetImageFromArray(ss_array)
            out_nifti.CopyInformation(nifti)

            WriteImage(out_nifti, SS_mask)

       
    else:
        for i in range(n):
            preprocessed_file = rows[i][pipeline_key]
            SS_file = preprocessed_file.replace(".nii.gz", "_SS.nii.gz")
            SS_mask = preprocessed_file.replace(".nii.gz", "_SS_mask.nii.gz")

            command = f"mri_synthstrip -i {preprocessed_file} -o {SS_file} -m {SS_mask}"
            if verbose:
                print(command)

            result = run(command.split(" "), capture_output=True)

            try:
                result.check_returncode()

            except Exception:
                error = result.stderr
                print(error)
                e = open(f"{preprocessed_dir}/errors.txt", "a")
                e.write(f"{error}\n")
                setattr(study_df, "failed_preprocessing", True)
                return study_df

    if debug:
        if registration_target is None:
            main_SS_file = rows[0][pipeline_key].replace(".nii.gz", "_SS.nii.gz")
        else:
            main_SS_file = registration_target.replace(
                ".nii.gz", "_RAS_spacing_SS.nii.gz"
            )

    else:
        if registration_target is None:
            main_SS_file = rows[0][pipeline_key].replace(".nii.gz", "_SS.nii.gz")
        else:
            main_SS_file = registration_target.replace(".nii.gz", "_SS.nii.gz")


    study_SS_file = rows[0][pipeline_key]
    study_SS_mask_file = rows[0][pipeline_key].replace(".nii.gz", "_SS_mask.nii.gz")

    ### Register based on loose skullstrip
    for i in range(1, n):
        rows[i] = local_reg(
            rows[i], pipeline_key, study_SS_file, registration_model, verbose, debug
        )

    if registration_target is not None:
        rows = long_reg(
            rows,
            pipeline_key,
            main_SS_file,
            study_SS_mask_file,
            registration_model,
            verbose,
            debug,
        )
        study_SS_mask_file = study_SS_mask_file.replace(".nii.gz", "_longreg.nii.gz")
    study_SS_mask_array = np.round(GetArrayFromImage(ReadImage(study_SS_mask_file)))

    ### Bias correction
    if not skullstrip:
        foreground_file = rows[0][pipeline_key].replace(
            ".nii.gz", "_foreground_mask.nii.gz"
        )
        nifti = ReadImage(rows[0][pipeline_key])
        foreground = RescaleIntensity(nifti, 0, 100)

        if pre_skullstripped:
            foreground_array = 1 - (GetArrayFromImage(foreground) == 0).astype(int)

        else:
            foreground_array = (GetArrayFromImage(foreground) > 15).astype(int)
            foreground_array = fill_foreground_mask(foreground_array)

        foreground = GetImageFromArray(foreground_array)
        foreground.CopyInformation(nifti)
        WriteImage(foreground, foreground_file)

    for i in range(n):
        preprocessed_file = rows[i][pipeline_key]

        if debug:
            output_file = preprocessed_file.replace(".nii.gz", "_N4.nii.gz")
            rows[i][pipeline_key] = output_file

        else:
            output_file = preprocessed_file

        raw_input = ReadImage(preprocessed_file, sitkFloat32)
        array = GetArrayFromImage(raw_input)
        array[array < 0] = 0.1
        n4_input = GetImageFromArray(array)
        n4_input.CopyInformation(raw_input)

        if skullstrip:
            n4_mask = ReadImage(study_SS_mask_file, sitkUInt8)

        else:
            n4_mask = foreground

        n4_mask = Cast(n4_mask, sitkUInt8)

        bias_corrector = N4BiasFieldCorrectionImageFilter()
        n4_corrected = bias_corrector.Execute(n4_input, n4_mask)

        sitk_im_cache[output_file] = n4_corrected

        if debug:
            WriteImage(n4_corrected, output_file)

    ### appy final skullmask if skullstripping
    if skullstrip:
        for i in range(n):
            preprocessed_file = rows[i][pipeline_key]

            if debug:
                output_file = preprocessed_file.replace(
                    ".nii.gz", "_skullstripped.nii.gz"
                )
                rows[i][pipeline_key] = output_file

            else:
                output_file = preprocessed_file

            nifti = sitk_im_cache[preprocessed_file]
            array = GetArrayFromImage(nifti)

            array = array * study_SS_mask_array

            output_nifti = GetImageFromArray(array)
            output_nifti.CopyInformation(nifti)

            sitk_im_cache[output_file] = output_nifti

            if debug:
                WriteImage(output_nifti, output_file)

            if "seg" in rows[i] and not pd.isna(rows[i]["seg"]):
                preprocessed_seg = rows[i][f"{pipeline_key}_seg"]

                if debug:
                    output_seg = preprocessed_seg.replace(
                        ".nii.gz", "_skullstripped.nii.gz"
                    )
                    rows[i][f"{pipeline_key}_seg"] = output_seg

                else:
                    output_seg = preprocessed_seg

                nifti = ReadImage(preprocessed_seg)
                array = GetArrayFromImage(nifti)

                array = array * study_SS_mask_array

                output_nifti = GetImageFromArray(array)
                output_nifti.CopyInformation(nifti)

                sitk_im_cache[output_seg] = output_nifti

                if debug:
                    WriteImage(output_nifti, output_seg)

    ### Normalization
    for i in range(n):
        preprocessed_file = rows[i][pipeline_key]

        if debug:
            output_file = preprocessed_file.replace(".nii.gz", "_norm.nii.gz")
            rows[i][pipeline_key] = output_file

        else:
            output_file = preprocessed_file

        nifti = sitk_im_cache[preprocessed_file]
        array = GetArrayFromImage(nifti)
        masked_input_array = np.ma.masked_where(study_SS_mask_array == 0, array)
        mean = np.ma.mean(masked_input_array)
        std = np.ma.std(masked_input_array)
        array = (array - mean) / (std + 1e-6)

        output_nifti = GetImageFromArray(array)
        output_nifti.CopyInformation(nifti)

        sitk_im_cache[output_file] = output_nifti

        if debug:
            WriteImage(output_nifti, output_file)

        ### set background back to 0 for easy foreground cropping
        if skullstrip:
            array = array * study_SS_mask_array

        else:
            array = array * foreground_array

        preprocessed_file = rows[i][pipeline_key]

        if debug:
            output_file = preprocessed_file.replace(".nii.gz", "_0background.nii.gz")
            rows[i][pipeline_key] = output_file

        else:
            output_file = preprocessed_file

        output_nifti = GetImageFromArray(array)
        output_nifti.CopyInformation(nifti)

        sitk_im_cache[output_file] = output_nifti

        if debug:
            WriteImage(output_nifti, output_file)

    ### Orientation
    if orientation != "RAS":
        orienter.SetDesiredCoordinateOrientation(orientation)

        for i in range(n):
            preprocessed_file = rows[i][pipeline_key]

            if debug:
                output_file = preprocessed_file.replace(
                    ".nii.gz", f"_{orientation}.nii.gz"
                )
                rows[i][pipeline_key] = output_file

            else:
                output_file = preprocessed_file

            nifti = sitk_im_cache[preprocessed_file]
            output_nifti = orienter.Execute(nifti)

            sitk_im_cache[output_file] = output_nifti

            if debug:
                WriteImage(output_nifti, output_file)

            if "seg" in rows[i] and not pd.isna(rows[i]["seg"]):
                preprocessed_seg = rows[i][f"{pipeline_key}_seg"]

                if debug:
                    output_seg = preprocessed_seg.replace(
                        ".nii.gz", f"_{orientation}.nii.gz"
                    )
                    rows[i][f"{pipeline_key}_seg"] = output_seg

                else:
                    output_seg = preprocessed_seg

                nifti = sitk_im_cache[preprocessed_seg]
                output_nifti = orienter.Execute(nifti)

                sitk_im_cache[output_seg] = output_nifti

                if debug:
                    WriteImage(output_nifti, output_seg)

    if not debug:
        for i in range(n):
            preprocessed_file = rows[i][pipeline_key]

            WriteImage(sitk_im_cache[preprocessed_file], preprocessed_file)


    ### copy metadata
    preprocessing_args = {
        "preprocessed_dir": str(preprocessed_dir),
        "pipeline_key": pipeline_key,
        "registration_target": registration_target,
        "registration_model": registration_model,
        "orientation": orientation,
        "spacing": spacing,
        "skullstrip": skullstrip,
        "pre_skullstripped": pre_skullstripped,
        "binarize_seg": binarize_seg,
    }

    for row in rows:
        copy_metadata(row, preprocessing_args)

    preprocessed_df = pd.DataFrame(rows)
    out_df = pd.merge(study_df, preprocessed_df, "outer")

    setattr(out_df, "failed_preprocessing", False)

    return out_df


def preprocess_patient(
    patient_df: pd.DataFrame,
    preprocessed_dir: Path | str,
    pipeline_key: str = "preprocessed",
    registration_key: str = "T1Post",
    longitudinal_registration: bool = False,
    registration_model: Literal["rigid", "affine", "joint", "deform"] = "affine",
    orientation: str = "RAS",
    spacing: str = "1,1,1",
    skullstrip: bool = True,
    pre_skullstripped: bool = False,
    binarize_seg: bool = False,
    verbose: bool = False,
    source_software: bool = True,
    check_columns: bool = True,
    debug: bool = False,
):
    """
    Preprocess all of the studies for a patient in a DataFrame.

    Parameters
    __________
    patient_df: pd.DataFrame
        A DataFrame containing nifti location and information required for the output file names
        for a single patient. It must contain the columns: 'nifti', 'Anon_PatientID', 'Anon_StudyID',
        'StudyInstanceUID', 'SeriesInstanceUID', 'NormalizedSeriesDescription', and 'SeriesType'.
    preprocessed_dir: Path
        The directory that will contain the preprocessed NIfTI files.
    pipeline_key: str
        The key that will be added to the DataFrame to indicate the new locations of preprocessed files.
        Defaults to 'preprocessed'.
    registration_key: str
        The value that will be used to select the fixed image during registration. This should correspond
        to a value within the 'NormalizedSeriesDescription' column in the csv. If you have segmentation
        files in your data. They should correspond to this same series. Defaults to 'T1Post'.
    longitudinal_registration: bool
        Whether to register all of the subsequent studies for a patient to the first study. Defaults to
        False.
    registration_model: str
        The synthmorph model that will be used to perform registration. Choices are: 'rigid', 'affine', 'joint',
        and 'deform'. Defaults to 'affine'.
    orientation: str
        The orientation standard that you wish to set for preprocessed data. Defaults to 'RAS'."
    spacing: str
        A comma delimited list indicating the desired spacing of preprocessed data. Measurements
        are in mm. Defaults to '1,1,1'.
    skullstrip: bool
        Whether to apply skullstripping to preprocessed data. Skullstripping will be applied by default.
    pre_skullstripped: bool
        Whether the input data is already skullstripped. Skullstripping will not be applied if specified.
    binarize_seg: bool
        Whether to binarize segmentations. Not recommended for multi-class labels. Binarization is not
        applied by default.
    verbose: bool
        Whether to print additional information related like commands and their arguments are printed.
    source_software: bool
        Whether to call `source_external_software` to add software required for preprocessing. Defaults
        to True.
    check_columns: bool
        Whether to check `study_df` for required columns. Defaults to True.
    debug: bool
        Whether to run in 'debug mode' where each step is saved with an individual name and intermediate
        files are not deleted. Dafaults to False.

    Returns
    _______
    pd.DataFrame:
        A Dataframe with added column f'{pipeline_key}' and optionally f'{pipeline_key}_seg' to indicate
        the locations of the preprocessing outputs.

    """

    if source_software:
        source_external_software()

    if check_columns:
        required_columns = [
            "nifti",
            "Anon_PatientID",
            "Anon_StudyID",
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "NormalizedSeriesDescription",
            "SeriesType",
        ]
        optional_columns = ["seg"]

        check_required_columns(patient_df, required_columns, optional_columns)

    if patient_df.shape[0] == 0:
        return patient_df

    preprocessed_dir = Path(preprocessed_dir)

    study_uids = patient_df["StudyInstanceUID"].unique()

    preprocessed_dfs = []

    study_df = patient_df[patient_df["StudyInstanceUID"] == study_uids[0]].copy()

    if pre_skullstripped:
        skullstrip = False

    preprocessed_dfs.append(
        preprocess_study(
            study_df=study_df,
            preprocessed_dir=preprocessed_dir,
            pipeline_key=pipeline_key,
            registration_key=registration_key,
            registration_target=None,
            registration_model=registration_model,
            orientation=orientation,
            spacing=spacing,
            skullstrip=skullstrip,
            pre_skullstripped=pre_skullstripped,
            binarize_seg=binarize_seg,
            verbose=verbose,
            source_software=False,
            check_columns=False,
            debug=debug,
        )
    )

    if getattr(preprocessed_dfs[0], "failed_preprocessing", False):
        # remove failed preprocessed studies
        anon_patientID = patient_df.loc[patient_df.index[0], "Anon_PatientID"]
        anon_studyID = study_df.loc[study_df.index[0], "Anon_StudyID"]
        study_dir = preprocessed_dir / anon_patientID / anon_studyID

        if study_dir.exists():
            shutil.rmtree(study_dir)

        if patient_df.shape[0] > 1:
            patient_df = patient_df.loc[1:, :]
            preprocess_patient(
                patient_df=patient_df,
                preprocessed_dir=preprocessed_dir,
                pipeline_key=pipeline_key,
                registration_key=registration_key,
                longitudinal_registration=longitudinal_registration,
                registration_model=registration_model,
                orientation=orientation,
                spacing=spacing,
                skullstrip=skullstrip,
                pre_skullstripped=pre_skullstripped,
                binarize_seg=binarize_seg,
                verbose=verbose,
                source_software=source_software,
                check_columns=check_columns,
                debug=debug,
            )

        else:
            return patient_df

    else:
        if len(study_uids) > 1:
            if longitudinal_registration:
                reg_sorted = (
                    preprocessed_dfs[0]
                    .sort_values(
                        ["NormalizedSeriesDescription"],
                        key=lambda x: (x != registration_key).astype(int),
                    )
                    .reset_index(drop=True)
                )

                if debug:
                    output_dir = os.path.dirname(reg_sorted[pipeline_key][0])
                    base_name = os.path.basename(reg_sorted["nifti"][0])
                    registration_target = f"{output_dir}/{base_name}"

                else:
                    registration_target = reg_sorted[pipeline_key][0]

                for study_uid in study_uids[1:]:
                    study_df = patient_df[
                        patient_df["StudyInstanceUID"] == study_uid
                    ].copy()

                    preprocessed_dfs.append(
                        preprocess_study(
                            study_df=study_df,
                            preprocessed_dir=preprocessed_dir,
                            pipeline_key=pipeline_key,
                            registration_key=registration_key,
                            registration_target=registration_target,
                            registration_model=registration_model,
                            orientation=orientation,
                            spacing=spacing,
                            skullstrip=skullstrip,
                            pre_skullstripped=pre_skullstripped,
                            binarize_seg=binarize_seg,
                            verbose=verbose,
                            source_software=False,
                            check_columns=False,
                            debug=debug,
                        )
                    )

            else:
                for study_uid in study_uids[1:]:
                    study_df = patient_df[
                        patient_df["StudyInstanceUID"] == study_uid
                    ].copy()

                    preprocessed_dfs.append(
                        preprocess_study(
                            study_df=study_df,
                            preprocessed_dir=preprocessed_dir,
                            pipeline_key=pipeline_key,
                            registration_key=registration_key,
                            registration_target=None,
                            registration_model=registration_model,
                            orientation=orientation,
                            spacing=spacing,
                            skullstrip=skullstrip,
                            pre_skullstripped=pre_skullstripped,
                            verbose=verbose,
                            source_software=False,
                            check_columns=False,
                            debug=debug,
                        )
                    )

        # clear extra files
        anon_patientID = patient_df.loc[patient_df.index[0], "Anon_PatientID"]
        patient_dir = preprocessed_dir / anon_patientID

        out_df = pd.concat(preprocessed_dfs, ignore_index=True)

        if not debug:
            extra_files = (
                list(patient_dir.glob("**/*SS.nii.gz"))
                + list(patient_dir.glob("**/*mask.nii.gz"))
                + list(patient_dir.glob("**/*longreg.nii.gz"))
                + list(patient_dir.glob("**/*.mgz"))
                + list(patient_dir.glob("**/*.m3z"))
                + list(patient_dir.glob("**/*.txt"))
            )

            print("......Clearing unnecessary files......")
            for file in extra_files:
                os.remove(file)

        print(f"Finished preprocessing {anon_patientID}:")
        print(out_df)
        return out_df


def preprocess_from_csv(
    csv: Path | str,
    preprocessed_dir: Path | str,
    patients: Sequence[str] | None = None,
    pipeline_key: str = "preprocessed",
    registration_key: str = "T1Post",
    longitudinal_registration: bool = False,
    registration_model: Literal["rigid", "affine", "joint", "deform"] = "affine",
    orientation: str = "RAS",
    spacing: str = "1,1,1",
    skullstrip: bool = True,
    pre_skullstripped: bool = False,
    binarize_seg: bool = False,
    cpus: int = 0,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Preprocess all of the studies for a patient in a DataFrame.

    Parameters
    __________
    csv: Path | str
        The path to a CSV containing an entire dataset. It must contain the following columns:  'nifti',
        'Anon_PatientID', 'Anon_StudyID', 'StudyInstanceUID', 'SeriesInstanceUID', 'NormalizedSeriesDescription',
        and 'SeriesType'.
    preprocessed_dir: Path
        The directory that will contain the preprocessed NIfTI files.
    patients: Sequece[str] | None
        A sequence of patients to select from the 'Anon_PatientID' column of the CSV. If 'None' is provided,
        all patients will be preprocessed.
    pipeline_key: str
        The key that will be added to the DataFrame to indicate the new locations of preprocessed files.
        Defaults to 'preprocessed'.
    registration_key: str
        The value that will be used to select the fixed image during registration. This should correspond
        to a value within the 'NormalizedSeriesDescription' column in the csv. If you have segmentation
        files in your data. They should correspond to this same series. Defaults to 'T1Post'.
    longitudinal_registration: bool
        Whether to register all of the subsequent studies for a patient to the first study. Defaults to
        False.
    registration_model: str
        The synthmorph model that will be used to perform registration. Choices are: 'rigid', 'affine', 'joint',
        and 'deform'. Defaults to 'affine'.
    orientation: str
        The orientation standard that you wish to set for preprocessed data. Defaults to 'RAS'."
    spacing: str
        A comma delimited list indicating the desired spacing of preprocessed data. Measurements
        are in mm. Defaults to '1,1,1'.
    skullstrip: bool
        Whether to apply skullstripping to preprocessed data. Skullstripping will be applied by default.
    pre_skullstripped: bool
        Whether the input data is already skullstripped. Skullstripping will not be applied if specified.
    binarize_seg: bool
        Whether to binarize segmentations. Not recommended for multi-class labels. Binarization is not
        applied by default.
    cpus: int
        Number of cpus to use for multiprocessing. Defaults to 1 (no multiprocessing).
    verbose: bool
        Whether to print additional information such as individual commands and their arguments. Defaults to False.

    Returns
    _______
    pd.DataFrame:
        A Dataframe with added column f'{pipeline_key}' and optionally f'{pipeline_key}_seg' to indicate
        the locations of the preprocessing outputs. This function will also overwrite the input CSV with
        this DataFrame.
    """

    source_external_software()

    df = pd.read_csv(csv, dtype=str)

    if pipeline_key in df.keys():
        df = df.drop(columns=pipeline_key)
        if f"{pipeline_key}_seg" in df.keys():
            df = df.drop(columns=f"{pipeline_key}_seg")

    df.to_csv(csv, index=False)

    required_columns = [
        "nifti",
        "Anon_PatientID",
        "Anon_StudyID",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "NormalizedSeriesDescription",
        "SeriesType",
    ]
    optional_columns = ["seg"]

    check_required_columns(df, required_columns, optional_columns)

    preprocessed_dir = Path(preprocessed_dir)

    df = df.drop_duplicates(subset="SeriesInstanceUID").reset_index(drop=True)

    filtered_df = df.copy().dropna(subset="nifti")
    
    if patients is None:
        patients = filtered_df["Anon_PatientID"].unique()

    if pre_skullstripped:
        skullstrip = False

    kwargs_list = [
        {
            "patient_df": filtered_df[filtered_df["Anon_PatientID"] == patient].copy(),
            "preprocessed_dir": preprocessed_dir,
            "pipeline_key": pipeline_key,
            "registration_key": registration_key,
            "longitudinal_registration": longitudinal_registration,
            "registration_model": registration_model,
            "orientation": orientation,
            "spacing": spacing,
            "skullstrip": skullstrip,
            "pre_skullstripped": pre_skullstripped,
            "binarize_seg": binarize_seg,
            "verbose": verbose,
            "source_software": False,
            "check_columns": False,
        }
        for patient in patients
    ]

    with tqdm(
        total=len(kwargs_list), desc="Preprocessing patients"
    ) as pbar, ThreadPoolExecutor(cpus if cpus >= 1 else 1) as executor:
        futures = [
            executor.submit(preprocess_patient, **kwargs) for kwargs in kwargs_list
        ]
        for future in as_completed(futures):
            preprocessed_df = future.result()
            df = pd.read_csv(csv, dtype=str)
            df = pd.merge(df, preprocessed_df, how="outer")
            df = (
                df.drop_duplicates(subset="SeriesInstanceUID")
                .sort_values(["Anon_PatientID", "Anon_StudyID"])
                .reset_index(drop=True)
            )
            df.to_csv(csv, index=False)
            pbar.update(1)

    df = pd.read_csv(csv, dtype=str)
    return df


def debug_from_csv(
    csv: Path | str,
    preprocessed_dir: Path | str,
    patients: Sequence[str] | None = None,
    pipeline_key: str = "debug",
    registration_key: str = "T1Post",
    longitudinal_registration: bool = False,
    registration_model: Literal["rigid", "affine", "joint", "deform"] = "affine",
    orientation: str = "RAS",
    spacing: str = "1,1,1",
    skullstrip: bool = True,
    pre_skullstripped: bool = False,
    binarize_seg: bool = False,
    cpus: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Preprocess all of the studies for a patient in a DataFrame.

    Parameters
    __________
    csv: Path | str
        The path to a CSV containing an entire dataset. It must contain the following columns:  'nifti',
        'Anon_PatientID', 'Anon_StudyID', 'StudyInstanceUID', 'SeriesInstanceUID', 'NormalizedSeriesDescription',
        and 'SeriesType'.
    preprocessed_dir: Path
        The directory that will contain the preprocessed NIfTI files.
    patients: Sequece[str] | None
        A sequence of patients to select from the 'Anon_PatientID' column of the CSV. If 'None' is provided,
        all patients will be preprocessed.
    pipeline_key: str
        The key that will be added to the DataFrame to indicate the new locations of preprocessed files.
        Defaults to 'debug'.
    registration_key: str
        The value that will be used to select the fixed image during registration. This should correspond
        to a value within the 'NormalizedSeriesDescription' column in the csv. If you have segmentation
        files in your data. They should correspond to this same series. Defaults to 'T1Post'.
    longitudinal_registration: bool
        Whether to register all of the subsequent studies for a patient to the first study. Defaults to
        False.
    registration_model: str
        The synthmorph model that will be used to perform registration. Choices are: 'rigid', 'affine', 'joint',
        and 'deform'. Defaults to 'affine'.
    orientation: str
        The orientation standard that you wish to set for preprocessed data. Defaults to 'RAS'."
    spacing: str
        A comma delimited list indicating the desired spacing of preprocessed data. Measurements
        are in mm. Defaults to '1,1,1'.
    skullstrip: bool
        Whether to apply skullstripping to preprocessed data. Skullstripping will be applied by default.
    pre_skullstripped: bool
        Whether the input data is already skullstripped. Skullstripping will not be applied if specified.
    binarize_seg: bool
        Whether to binarize segmentations. Not recommended for multi-class labels. Binarization is not
        applied by default.
    cpus: int
        Number of cpus to use for multiprocessing. Defaults to 1 (no multiprocessing).
    verbose: bool
        Whether to print additional information such as individual commands and their arguments. Defaults to False.

    Returns
    _______
    pd.DataFrame:
        A Dataframe with added column f'{pipeline_key}' and optionally f'{pipeline_key}_seg' to indicate
        the locations of the preprocessing outputs. This function will also overwrite the input CSV with
        this DataFrame.
    """

    source_external_software()

    df = pd.read_csv(csv, dtype=str)
    csv = preprocessed_dir / "debug.csv"

    if pipeline_key in df.keys():
        df = df.drop(columns=pipeline_key)
        if f"{pipeline_key}_seg" in df.keys():
            df = df.drop(columns=f"{pipeline_key}_seg")

    df.to_csv(csv, index=False)
   
    required_columns = [
        "nifti",
        "Anon_PatientID",
        "Anon_StudyID",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "NormalizedSeriesDescription",
        "SeriesType",
    ]
    optional_columns = ["seg"]

    check_required_columns(df, required_columns, optional_columns)

    preprocessed_dir = Path(preprocessed_dir)

    df = df.drop_duplicates(subset="SeriesInstanceUID").reset_index(drop=True)

    filtered_df = df.copy().dropna(subset="nifti")

    if patients is None:
        patients = filtered_df["Anon_PatientID"].unique()

    if pre_skullstripped:
        skullstrip = False

    kwargs_list = [
        {
            "patient_df": filtered_df[filtered_df["Anon_PatientID"] == patient].copy(),
            "preprocessed_dir": preprocessed_dir,
            "pipeline_key": pipeline_key,
            "registration_key": registration_key,
            "longitudinal_registration": longitudinal_registration,
            "registration_model": registration_model,
            "orientation": orientation,
            "spacing": spacing,
            "skullstrip": skullstrip,
            "pre_skullstripped": pre_skullstripped,
            "binarize_seg": binarize_seg,
            "verbose": verbose,
            "source_software": False,
            "check_columns": False,
            "debug": True,
        }
        for patient in patients
    ]

    with tqdm(
        total=len(kwargs_list), desc="Preprocessing patients"
    ) as pbar, ThreadPoolExecutor(cpus if cpus >= 1 else 1) as executor:
        futures = [
            executor.submit(preprocess_patient, **kwargs) for kwargs in kwargs_list
        ]
        for future in as_completed(futures):
            preprocessed_df = future.result()
            df = (
                pd.read_csv(csv, dtype=str)
                .drop_duplicates(subset="SeriesInstanceUID")
                .reset_index(drop=True)
            )
            df = pd.merge(df, preprocessed_df, how="outer")
            df = (
                df.drop_duplicates(subset="SeriesInstanceUID")
                .sort_values(["Anon_PatientID", "Anon_StudyID"])
                .reset_index(drop=True)
            )
            df.to_csv(csv, index=False)
            pbar.update(1)

    df = (
        pd.read_csv(csv, dtype=str)
        .drop_duplicates(subset="SeriesInstanceUID")
        .sort_values(["Anon_PatientID", "Anon_StudyID"])
        .reset_index(drop=True)
    )
    df.to_csv(csv, index=False)
    return df
