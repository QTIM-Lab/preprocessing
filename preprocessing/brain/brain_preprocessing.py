"""
The `brain_preprocessing` module defines the tools for processing MRI data
with a pipeline designed specifically for brain data.

Public Functions
________________
preprocess_study
    Preprocess a single study from a DataFrame.

preprocess_patient
    Preprocess all of the studies for a patient in a DataFrame.

preprocess_from_csv
    Preprocess all of the studies within a dataset.
"""
import os
import pandas as pd
import numpy as np
import json
import warnings

from SimpleITK import (
    DICOMOrientImageFilter,
    sitkLinear,
    sitkNearestNeighbor,
    ReadImage,
    Resample,
    ResampleImageFilter,
    Image,
    WriteImage,
    GetArrayFromImage,
    GetImageFromArray,
    OtsuThresholdImageFilter,
    N4BiasFieldCorrectionImageFilter,
    sitkFloat32,
    sitkUInt8,
    Cast,
)

from pathlib import Path
from tqdm import tqdm
from preprocessing.utils import check_required_columns, cpu_adjust

from preprocessing.synthmorph import synthmorph_registration
from .synthstrip import synthstrip_skullstrip
from typing import Sequence, List, Literal, Dict, Any, Tuple
from scipy.ndimage import binary_fill_holes
from cc3d import connected_components
from concurrent.futures import ProcessPoolExecutor, as_completed


def copy_metadata(row: Dict[str, Any], preprocessing_args: Dict[str, Any]) -> None:
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
    fixed_image_path: str,
    moving_image_path: str,
    sitk_im_cache: Dict[str, Image],
    interp_method: Literal["linear", "nearest"] = "linear",
    verbose: bool = False,
) -> Tuple[bool, Dict[str, Image]]:
    """
    Verify the quality of registrations by checking for consistency in array shapes and
    affines. In cases of failure, the moving image will be resampled to the fixed image
    but alignment is not guaranteed.

    Parameters
    __________
    fixed_image_path: str
        The path to the fixed image, which must be a key within `sitk_im_cache`.

    moving_image_path: str
        The path to the moving image, which must be a key within `sitk_im_cache`.

    sitk_im_cache: Dict[str, Image]
        The cache used to store intermediate files within the registration pipeline following this
        format: {path: Image}.

    interp_method: Literal["linear", "nearest"]
        The interpolation method to use if the images needs to be resampled. Defaults to "linear".

    verbose: bool
        Whether to print additional information related like commands and their arguments are printed.

    Returns
    _______
    good_registrations: bool
        Whether the registration was of good quality and did not required resampling.

    sitk_im_cache: Dict[str, Image]
        A potentially updated version of the input `sitk_im_cache`, which contains the resampled images
        if applicable.
    """
    if verbose:
        print(f"{moving_image_path} is being checked against {fixed_image_path}")

    fixed_image = sitk_im_cache[fixed_image_path]
    moving_image = sitk_im_cache[moving_image_path]
    same_shape = fixed_image.GetSize() == moving_image.GetSize()

    close_affines = (
        np.allclose(fixed_image.GetDirection(), moving_image.GetDirection())
        and np.allclose(fixed_image.GetSpacing(), moving_image.GetSpacing())
        and np.allclose(fixed_image.GetOrigin(), moving_image.GetOrigin())
    )

    if not same_shape or not close_affines:
        if not same_shape:
            print(
                f"Shapes do not match: {fixed_image.GetSize()} {moving_image.GetSize()}"
            )
        if not close_affines:
            print("Affines do not match")
        print(f"Resampling {moving_image_path} to {fixed_image_path}")
        resampler = ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        if interp_method == "linear":
            resampler.SetInterpolator(sitkLinear)
        elif interp_method == "nearest":
            resampler.SetInterpolator(sitkNearestNeighbor)

        resampled_image = resampler.Execute(moving_image)

        sitk_im_cache[moving_image_path] = resampled_image

        return False, sitk_im_cache

    else:
        if verbose:
            print(f"{moving_image_path} does not require resampling")
        return True, sitk_im_cache


def local_reg(
    row: Dict[str, Any],
    pipeline_key: str,
    fixed_image_path: str,
    sitk_im_cache: Dict[str, Image],
    model: Literal["rigid", "affine", "joint", "deform"] = "affine",
    verbose: bool = False,
    debug: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Image]]:
    """
    Perform registration on a series using Synthmorph. Meant for registration of images within the same study.

    Parameters
    __________
    row: Dict[str, Any]
        A dictionary representing a series row from the study DataFrame / CSV.

    pipeline_key:
        The key that will be added to the DataFrame to indicate the new locations of preprocessed files.
        Defaults to 'preprocessed'.

    fixed_image_path: str
        The path to the fixed image, which must be a key within `sitk_im_cache`.

    sitk_im_cache: Dict[str, Image]
        The cache used to store intermediate files within the registration pipeline following this
        format: {path: Image}.

    model: str
        The Synthmorph model that will be used to perform registration. Choices are: 'rigid', 'affine', 'joint',
        and 'deform'. Defaults to 'affine'.

    verbose: bool
        Whether to print additional information related like commands and their arguments are printed. Defaults
        to False.

    debug: bool
        Whether to run in 'debug mode' where each step is saved with an individual name and intermediate
        files are not deleted. Dafaults to False.

    Returns
    _______
    row: Dict[str, Any]
        An updated version of the input `row`, which contains the updated path for the moved image.

    sitk_im_cache: Dict[str, Image]
        An updated version of the input `sitk_im_cache`, which contains the moved image.
    """

    preprocessed_file = row[pipeline_key]

    if debug:
        output_file = preprocessed_file.replace(".nii.gz", "_localreg.nii.gz")
        row[pipeline_key] = output_file

    else:
        output_file = preprocessed_file

    moving_image_path = preprocessed_file.replace(".nii.gz", "_SS.nii.gz")

    accompanying_images = [{"moving": preprocessed_file, "moved": output_file}]

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
                "moved": output_seg,
                "interp_method": "nearest",
            }
        )

    sitk_im_cache = synthmorph_registration(
        sitk_im_cache[moving_image_path],
        sitk_im_cache[fixed_image_path],
        accompanying_images=accompanying_images,
        m=model,
        sitk_im_cache=sitk_im_cache,
        accompanying_in_cache=True,
    )
    if verbose:
        print(
            f"Registered files generated to {[d['moved'] for d in accompanying_images]}"
        )

    good_registrations, sitk_im_cache = verify_reg(
        fixed_image_path, output_file, sitk_im_cache, verbose=verbose
    )

    if not good_registrations:
        if verbose:
            print(
                "Registered images do not share the same affine or shape and require resampling."
            )

        if len(accompanying_images) > 1:
            for accompanying_image in accompanying_images:
                moving_image_path = accompanying_image["moved"]
                interp_method = accompanying_image.get("interp_method", "linear")

                sitk_im_cache = verify_reg(
                    fixed_image_path,
                    moving_image_path,
                    sitk_im_cache,
                    interp_method=interp_method,
                    verbose=verbose,
                )[1]

            if verbose:
                print("Resampling completed.")

    return row, sitk_im_cache


def long_reg(
    rows: List[Dict[str, Any]],
    pipeline_key: str,
    fixed_image_path: str,
    study_SS_mask_file: str,
    sitk_im_cache: Dict[str, Image],
    model: Literal["rigid", "affine", "joint", "deform"] = "affine",
    verbose: bool = False,
    debug: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Image]]:
    """
    Perform longitudinal registration on a study using synthmorph.

    Parameters
    __________
    row: List[Dict[str, Any]]
        A list of dictionaries with each representing a series row from the study DataFrame / CSV.
    pipeline_key:
        The key that will be added to the DataFrame to indicate the new locations of preprocessed files.
        Defaults to 'preprocessed'.
    fixed_image_path: str
        The path to the fixed image, which must be a key within `sitk_im_cache`.
    study_SS_mask_file: str
        The path to the skullstrip mask chosen for the study, which must be a key within `sitk_im_cache`.
    sitk_im_cache: Dict[str, Image]
        The cache used to store intermediate files within the registration pipeline following this
        format: {path: Image}.
    model: str
        The Synthmorph model that will be used to perform registration. Choices are: 'rigid', 'affine', 'joint',
        and 'deform'. Defaults to 'affine'.
    verbose: bool
        Whether to print additional information related like commands and their arguments are printed. Defaults
        to False.
    debug: bool
        Whether to run in 'debug mode' where each step is saved with an individual name and intermediate
        files are not deleted. Dafaults to False.

    Returns
    _______
    rows: List[Dict[str, Any]]
        An updated version of the input `rows`, which contains the updated paths for the moved images.
    sitk_im_cache: Dict[str, Image]
        An updated version of the input `sitk_im_cache`, which contains the moved images.
    """

    moving_image_path = rows[0][pipeline_key].replace(".nii.gz", "_SS.nii.gz")

    if debug:
        moved_image_path = moving_image_path.replace(".nii.gz", "_longreg.nii.gz")

    else:
        moved_image_path = moving_image_path

    accompanying_images = [
        {
            "moving": study_SS_mask_file,
            "moved": study_SS_mask_file.replace(".nii.gz", "_longreg.nii.gz"),
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

        accompanying_images.append({"moving": preprocessed_file, "moved": output_file})

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
                    "moved": output_seg,
                    "interp_method": "nearest",
                }
            )
        rows[i] = row

    sitk_im_cache = synthmorph_registration(
        sitk_im_cache[moving_image_path],
        sitk_im_cache[fixed_image_path],
        o=moved_image_path,
        sitk_im_cache=sitk_im_cache,
        accompanying_images=accompanying_images,
        accompanying_in_cache=True,
        m=model,
    )

    if verbose:
        print(
            f"Registered files generated to {[d['moved'] for d in accompanying_images]}"
        )

    good_registrations, sitk_im_cache = verify_reg(
        fixed_image_path, moved_image_path, sitk_im_cache, verbose=verbose
    )

    if not good_registrations:
        if verbose:
            print(
                "Registered images do not share the same affine or shape and require resampling."
            )

        if len(accompanying_images) > 1:
            for accompanying_image in accompanying_images:
                moving_image_path = accompanying_image["moved"]
                interp_method = accompanying_image.get("interp_method", "linear")

                sitk_im_cache = verify_reg(
                    fixed_image_path,
                    moving_image_path,
                    sitk_im_cache,
                    interp_method=interp_method,
                    verbose=verbose,
                )[1]

            if verbose:
                print("Resampling completed.")

    return rows, sitk_im_cache


def fill_foreground_mask(initial_foreground: np.ndarray) -> np.ndarray:
    """
    Fill the initial foreground mask so that it will include the entire foreground.

    Parameters
    __________
    initial_foreground: np.ndarray
        The initial foreground mask that represents the border of the foreground but is not filled.

    Returns
    _______
    foreground: np.ndarray
        The filled foreground mask.

    """
    shape = initial_foreground.shape
    foreground_cc = connected_components(initial_foreground)
    ccs, counts = np.unique(foreground_cc, return_counts=True)

    sorted_ccs = sorted(
        [(cc, count) if cc != 0 else (0, 0) for cc, count in zip(ccs, counts)],
        key=lambda x: x[1],
    )

    largest_cc = sorted_ccs[-1][0]

    foreground = (foreground_cc == largest_cc).astype(int)

    for z in range(shape[0]):
        foreground_slice = foreground[z, ...]
        if 1 not in np.unique(foreground_slice):
            continue
        for i in range(shape[1]):
            foreground_slice[i] = binary_fill_holes(foreground_slice[i]).astype(int)

        for j in range(shape[2]):
            foreground_slice[:, j] = binary_fill_holes(foreground_slice[:, j]).astype(
                int
            )

    return foreground.astype(int)


def preprocess_study(
    study_df: pd.DataFrame,
    preprocessed_dir: Path | str,
    pipeline_key: str,
    registration_key: str = "T1Post",
    registration_target: Path | str | None = None,
    registration_model: Literal["rigid", "affine", "joint", "deform"] = "affine",
    orientation: str = "RAS",
    spacing: Sequence[float | int] = [1, 1, 1],
    skullstrip: bool = True,
    pre_skullstripped: bool = False,
    binarize_seg: bool = False,
    verbose: bool = False,
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

    registration_target: Path | str | None
        The location of the file that will be used as the fixed image for the purposes of registration.

    orientation: str
        The orientation standard that you wish to set for preprocessed data. Defaults to 'RAI'."

    spacing: Sequence[float | int]
        A sequence of floats or ints indicating the desired spacing of preprocessed data. Measurements
        are in mm. Defaults to [1, 1, 1].

    skullstrip: bool
        Whether to apply skullstripping to preprocessed data. Skullstripping will be applied by default.

    pre_skullstripped: bool
        Whether the input data is already skullstripped. Skullstripping will not be applied if specified.

    binarize_seg: bool
        Whether to binarize segmentations. Not recommended for multi-class labels. Binarization is not
        applied by default.

    verbose: bool
        Whether to print additional information related like commands and their arguments are printed.

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

        rows[i][pipeline_key] = str(preprocessed_file)

        try:
            sitk_im_cache[str(preprocessed_file)] = ReadImage(input_file)

        except Exception:
            error = f"Could not create {preprocessed_file}"
            print(error)
            e = open(f"{preprocessed_dir}/errors.txt", "a")
            e.write(f"{error}\n")
            setattr(study_df, "failed_preprocessing", True)
            return study_df

        if "seg" in rows[i] and not pd.isna(rows[i]["seg"]):
            input_seg = rows[i]["seg"]
            preprocessed_seg = output_dir / os.path.basename(input_seg)

            rows[i][f"{pipeline_key}_seg"] = str(preprocessed_seg)

            try:
                sitk_im_cache[str(preprocessed_seg)] = ReadImage(input_seg)

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

                nifti = sitk_im_cache[preprocessed_seg]
                array = GetArrayFromImage(nifti)

                array = (array >= 1).astype(int)

                output_nifti = GetImageFromArray(array)
                output_nifti.CopyInformation(nifti)

                sitk_im_cache[output_seg] = output_nifti

                if verbose:
                    print(f"{preprocessed_seg} binarized")

    ### orientation
    orienter = DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation(orientation)

    for i in range(n):
        preprocessed_file = rows[i][pipeline_key]

        if debug:
            output_file = preprocessed_file.replace(".nii.gz", f"_{orientation}.nii.gz")
            rows[i][pipeline_key] = output_file

        else:
            output_file = preprocessed_file

        nifti = sitk_im_cache[preprocessed_file]
        output_nifti = orienter.Execute(nifti)

        sitk_im_cache[output_file] = output_nifti

        if verbose:
            print(f"{preprocessed_file} set to {orientation} orientation")

        if "seg" in rows[i] and not pd.isna(rows[i]["seg"]):
            preprocessed_seg = rows[i][f"{pipeline_key}_seg"]

            if debug:
                output_seg = preprocessed_seg.replace(".nii.gz", "_RAS.nii.gz")
                rows[i][f"{pipeline_key}_seg"] = output_seg

            else:
                output_seg = preprocessed_seg

            nifti = sitk_im_cache[preprocessed_seg]
            output_nifti = orienter.Execute(nifti)

            sitk_im_cache[output_seg] = output_nifti

            if verbose:
                print(f"{preprocessed_seg} set to {orientation} orientation")

    ### Spacing
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
            for osz, osp, ns in zip(original_size, original_spacing, spacing)
        ]
        output_nifti = Resample(
            nifti,
            new_size,
            interpolator=sitkLinear,
            outputOrigin=nifti.GetOrigin(),
            outputSpacing=spacing,
            outputDirection=nifti.GetDirection(),
        )

        sitk_im_cache[output_file] = output_nifti

        if verbose:
            print(f"{preprocessed_file} resampled to {spacing} spacing")

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
                for osz, osp, ns in zip(original_size, original_spacing, spacing)
            ]
            output_nifti = Resample(
                nifti,
                new_size,
                interpolator=sitkNearestNeighbor,
                outputOrigin=nifti.GetOrigin(),
                outputSpacing=spacing,
                outputDirection=nifti.GetDirection(),
            )

            sitk_im_cache[output_seg] = output_nifti

            if verbose:
                print(f"{preprocessed_seg} resampled to {spacing} spacing")

    ### Loose Skullstrip
    if pre_skullstripped:
        for i in range(n):
            preprocessed_file = rows[i][pipeline_key]
            SS_file = preprocessed_file.replace(".nii.gz", "_SS.nii.gz")
            SS_mask = preprocessed_file.replace(".nii.gz", "_SS_mask.nii.gz")

            nifti = sitk_im_cache[preprocessed_file]

            sitk_im_cache[SS_file] = nifti

            array = GetArrayFromImage(nifti)
            ss_array = 1 - (array == 0).astype(int)

            out_nifti = GetImageFromArray(ss_array)
            out_nifti.CopyInformation(nifti)

            sitk_im_cache[SS_mask] = out_nifti

            if verbose:
                print(f'"Loose" skullstrip mask obtained from {preprocessed_file}')

    else:
        for i in range(n):
            preprocessed_file = rows[i][pipeline_key]
            SS_file = preprocessed_file.replace(".nii.gz", "_SS.nii.gz")
            SS_mask = preprocessed_file.replace(".nii.gz", "_SS_mask.nii.gz")

            sitk_im_cache = synthstrip_skullstrip(
                image=sitk_im_cache[preprocessed_file],
                sitk_im_cache=sitk_im_cache,
                out=SS_file,
                m=SS_mask,
            )

            if verbose:
                print(f'"Loose" skullstrip mask obtained from {preprocessed_file}')

    if debug:
        if registration_target is not None:
            registration_target = str(registration_target)
            main_SS_file = registration_target.replace(
                ".nii.gz", "_RAS_spacing_SS.nii.gz"
            )

            if not Path(main_SS_file).exists():
                main_SS_file = registration_target

            sitk_im_cache[main_SS_file] = ReadImage(main_SS_file)

        else:
            main_SS_file = None

    else:
        if registration_target is not None:
            registration_target = str(registration_target)
            main_SS_file = registration_target.replace(".nii.gz", "_SS.nii.gz")

            if not Path(main_SS_file).exists():
                main_SS_file = registration_target

            sitk_im_cache[main_SS_file] = ReadImage(main_SS_file)

        else:
            main_SS_file = None

    study_SS_file = rows[0][pipeline_key].replace(".nii.gz", "_SS.nii.gz")
    study_SS_mask_file = rows[0][pipeline_key].replace(".nii.gz", "_SS_mask.nii.gz")

    ### Register based on loose skullstrip
    for i in range(1, n):
        rows[i], sitk_im_cache = local_reg(
            rows[i],
            pipeline_key,
            study_SS_file,
            sitk_im_cache,
            registration_model,
            verbose,
            debug,
        )

    if registration_target is not None:
        rows, sitk_im_cache = long_reg(
            rows,
            pipeline_key,
            main_SS_file,
            study_SS_mask_file,
            sitk_im_cache,
            registration_model,
            verbose,
            debug,
        )
        study_SS_mask_file = study_SS_mask_file.replace(".nii.gz", "_longreg.nii.gz")
    study_SS_mask_array = np.round(GetArrayFromImage(sitk_im_cache[study_SS_mask_file]))

    ### Bias correction
    if not skullstrip:
        foreground_file = rows[0][pipeline_key].replace(
            ".nii.gz", "_foreground_mask.nii.gz"
        )
        nifti = sitk_im_cache[rows[0][pipeline_key]]

        if pre_skullstripped:
            foreground_array = 1 - (GetArrayFromImage(nifti) == 0).astype(int)

        else:
            threshold_filter = OtsuThresholdImageFilter()
            threshold_filter.Execute(nifti)
            threshold = threshold_filter.GetThreshold()

            foreground_array = (GetArrayFromImage(nifti) >= threshold).astype(int)
            foreground_array = fill_foreground_mask(foreground_array)

        foreground = GetImageFromArray(foreground_array)
        foreground.CopyInformation(nifti)

        sitk_im_cache[foreground_file] = foreground

    for i in range(n):
        preprocessed_file = rows[i][pipeline_key]

        if debug:
            output_file = preprocessed_file.replace(".nii.gz", "_N4.nii.gz")
            rows[i][pipeline_key] = output_file

        else:
            output_file = preprocessed_file

        raw_input = Cast(sitk_im_cache[preprocessed_file], sitkFloat32)
        array = GetArrayFromImage(raw_input)
        array[array < 0] = 0.1
        n4_input = GetImageFromArray(array)
        n4_input.CopyInformation(raw_input)

        if skullstrip:
            n4_mask = Cast(sitk_im_cache[study_SS_mask_file], sitkUInt8)

        else:
            n4_mask = foreground

        n4_mask = Cast(n4_mask, sitkUInt8)

        bias_corrector = N4BiasFieldCorrectionImageFilter()
        n4_corrected = bias_corrector.Execute(n4_input, n4_mask)

        sitk_im_cache[output_file] = n4_corrected

        if verbose:
            print(f"{preprocessed_file} underwent N4 bias field correction")

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

            if verbose:
                print(f"Study skullstrip mask applied to {preprocessed_file}")

            if "seg" in rows[i] and not pd.isna(rows[i]["seg"]):
                preprocessed_seg = rows[i][f"{pipeline_key}_seg"]

                if debug:
                    output_seg = preprocessed_seg.replace(
                        ".nii.gz", "_skullstripped.nii.gz"
                    )
                    rows[i][f"{pipeline_key}_seg"] = output_seg

                else:
                    output_seg = preprocessed_seg

                nifti = sitk_im_cache[preprocessed_seg]
                array = GetArrayFromImage(nifti)

                array = array * study_SS_mask_array

                output_nifti = GetImageFromArray(array)
                output_nifti.CopyInformation(nifti)

                sitk_im_cache[output_seg] = output_nifti

                if verbose:
                    print(f"Study skullstrip mask applied to {preprocessed_seg}")

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

        if verbose:
            print(f"{preprocessed_file} intensity normalized")

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

        if verbose:
            print(f"{preprocessed_file} background set to 0")

    ### Write files:
    for k, v in sitk_im_cache.items():
        if k != main_SS_file:
            WriteImage(v, k)

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
    atlas_target: Path | str | None = None,
    registration_model: Literal["rigid", "affine", "joint", "deform"] = "affine",
    orientation: str = "RAS",
    spacing: Sequence[float | int] = [1, 1, 1],
    skullstrip: bool = True,
    pre_skullstripped: bool = False,
    binarize_seg: bool = False,
    verbose: bool = False,
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

    atlas_target: Path | str | None
        The path to an atlas file if using using an atlas for the registration. If provided, `longitudinal_registration`
        will be disabled.

    registration_model: str
        The Synthmorph model that will be used to perform registration. Choices are: 'rigid', 'affine', 'joint',
        and 'deform'. Defaults to 'affine'.

    orientation: str
        The orientation standard that you wish to set for preprocessed data. Defaults to 'RAS'.

    spacing: Sequence[float | int]
        A sequence of floats or ints indicating the desired spacing of preprocessed data. Measurements
        are in mm. Defaults to [1, 1, 1].

    skullstrip: bool
        Whether to apply skullstripping to preprocessed data. Skullstripping will be applied by default.

    pre_skullstripped: bool
        Whether the input data is already skullstripped. Skullstripping will not be applied if specified.

    binarize_seg: bool
        Whether to binarize segmentations. Not recommended for multi-class labels. Binarization is not
        applied by default.

    verbose: bool
        Whether to print additional information related like commands and their arguments are printed.

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

    if atlas_target:
        assert Path(atlas_target).exists(), f"{atlas_target} must exist if provided."

        if longitudinal_registration:
            longitudinal_registration = False
            warnings.warn(
                "Registering to an atlas is mutually exclusive with longitudinal registration. "
                "Your preference for longitudinal registration has been overwritten.",
                UserWarning,
            )

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
            registration_target=atlas_target,
            registration_model=registration_model,
            orientation=orientation,
            spacing=spacing,
            skullstrip=skullstrip,
            pre_skullstripped=pre_skullstripped,
            binarize_seg=binarize_seg,
            verbose=verbose,
            check_columns=False,
            debug=debug,
        )
    )

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
                        registration_target=atlas_target,
                        registration_model=registration_model,
                        orientation=orientation,
                        spacing=spacing,
                        skullstrip=skullstrip,
                        pre_skullstripped=pre_skullstripped,
                        verbose=verbose,
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
    atlas_target: Path | str | None = None,
    registration_model: Literal["rigid", "affine", "joint", "deform"] = "affine",
    orientation: str = "RAS",
    spacing: Sequence[float | int] = [1, 1, 1],
    skullstrip: bool = True,
    pre_skullstripped: bool = False,
    binarize_seg: bool = False,
    cpus: int = 1,
    verbose: bool = False,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Preprocess all of the studies within a dataset.

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

    atlas_target: Path | str | None
        The path to an atlas file if using using an atlas for the registration. If provided, `longitudinal_registration`
        will be disabled.

    registration_model: str
        The Synthmorph model that will be used to perform registration. Choices are: 'rigid', 'affine', 'joint',
        and 'deform'. Defaults to 'affine'.

    orientation: str
        The orientation standard that you wish to set for preprocessed data. Defaults to 'RAS'.

    spacing: Sequence[float | int]
        A sequence of floats or ints indicating the desired spacing of preprocessed data. Measurements
        are in mm. Defaults to [1, 1, 1].

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

    debug: bool
        Whether to run in debug mode. Each intermediate step will be saved using a suffix for differentiation.
        The input CSV will not be altered. Instead, a new copy will be saved to the output directory. Defaults
        to False.

    Returns
    _______
    pd.DataFrame:
        A Dataframe with added column f'{pipeline_key}' and optionally f'{pipeline_key}_seg' to indicate
        the locations of the preprocessing outputs. This function will also overwrite the input CSV with
        this DataFrame.
    """
    df = pd.read_csv(csv, dtype=str)

    if debug:
        csv = preprocessed_dir / "debug.csv"
        pipeline_key = "debug"

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
        patients = list(filtered_df["Anon_PatientID"].unique())

    if pre_skullstripped:
        skullstrip = False

    if atlas_target:
        assert Path(atlas_target).exists(), f"{atlas_target} must exist if provided."

        if longitudinal_registration:
            longitudinal_registration = False
            warnings.warn(
                "Registering to an atlas is mutually exclusive with longitudinal registration. "
                "Your preference for longitudinal registration has been overwritten.",
                UserWarning,
            )

    kwargs_list = [
        {
            "patient_df": filtered_df[filtered_df["Anon_PatientID"] == patient].copy(),
            "preprocessed_dir": preprocessed_dir,
            "pipeline_key": pipeline_key,
            "registration_key": registration_key,
            "longitudinal_registration": longitudinal_registration,
            "atlas_target": atlas_target,
            "registration_model": registration_model,
            "orientation": orientation,
            "spacing": spacing,
            "skullstrip": skullstrip,
            "pre_skullstripped": pre_skullstripped,
            "binarize_seg": binarize_seg,
            "verbose": verbose,
            "check_columns": False,
            "debug": debug,
        }
        for patient in patients
    ]

    cpus = cpu_adjust(max_process_mem=3e9, cpus=cpus)

    with tqdm(
        total=len(kwargs_list), desc="Preprocessing patients"
    ) as pbar, ProcessPoolExecutor(cpus if cpus >= 1 else 1) as executor:
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


__all__ = [
    "preprocess_study",
    "preprocess_patient",
    "preprocess_from_csv",
]
