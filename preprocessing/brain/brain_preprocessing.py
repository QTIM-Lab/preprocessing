### imports
import os
import shutil
import pandas as pd
import numpy as np
import multiprocessing
import json

from nibabel import save, load, as_closest_canonical, Nifti1Image
from nipype.interfaces.ants import N4BiasFieldCorrection
from pathlib import Path
from subprocess import run
from tqdm import tqdm
from preprocessing.utils import (
    source_external_software,
    check_required_columns,
    sitk_check,
)
from typing import Sequence


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
        error = f"{original_metafile} does not exist. New metafile will not be created"
        print(error)
        e = open(f"{preprocessing_args['preprocessed_dir']}/errors.txt", "a")
        e.write(f"{error}\n")

    if "seg" in row:
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
            error = (
                f"{original_metafile} does not exist. New metafile will not be created"
            )
            print(error)
            e = open(f"{preprocessing_args['preprocessed_dir']}/errors.txt", "a")
            e.write(f"{error}\n")


def registration(
    row,
    pipeline_key,
    fixed_image_path,
    main_SS_file,
    main_SS_mask_file,
    main_SS_mask_array,
    verbose,
    debug,
):
    preprocessed_file = row[pipeline_key]

    if debug:
        output_file = preprocessed_file.replace(".nii.gz", "_registered.nii.gz")
        row[pipeline_key] = output_file

    else:
        output_file = preprocessed_file

    moving_image_path = preprocessed_file.replace(".nii.gz", "_SS.nii.gz")
    moving_image_skullmask = preprocessed_file.replace(".nii.gz", "_SS_mask.nii.gz")
    transform_outfile = preprocessed_file.replace(".nii.gz", "_transform.tfm")
    sampling_percentage = 0.002
    x, y, z = main_SS_mask_array.shape

    command = (
        f"Slicer "
        f"--launch BRAINSFit --fixedVolume {fixed_image_path} "
        f"--movingVolume {moving_image_path} "
        "--transformType Rigid,ScaleVersor3D,ScaleSkewVersor3D,Affine "
        "--initializeTransformMode useCenterOfROIAlign "
        "--interpolationMode WindowedSinc "
        f"--outputTransform {transform_outfile} "
        f"--samplingPercentage {sampling_percentage} "
        f"--movingBinaryVolume {moving_image_skullmask} "
        f"--fixedBinaryVolume {main_SS_mask_file} "
        "--maskProcessingMode ROI"
    )

    if verbose:
        print(command)
    try:
        run(command.split(" "), capture_output=not verbose).check_returncode()

    except Exception as error:
        return row, error

    command = (
        f"Slicer --launch ResampleScalarVectorDWIVolume "
        f"{preprocessed_file} {output_file} -i bs -f {transform_outfile} -R {main_SS_file} -z {x},{y},{z}"
    )
    if verbose:
        print(command)
    try:
        run(command.split(" "), capture_output=not verbose).check_returncode()
        sitk_check(output_file)

    except Exception as error:
        return row, error

    if "seg" in row:
        preprocessed_seg = row[f"{pipeline_key}_seg"]

        if debug:
            output_seg = preprocessed_seg.replace(".nii.gz", "_resampled.nii.gz")
            row[f"{pipeline_key}_seg"]

        else:
            output_seg = preprocessed_seg

        command = (
            f"Slicer --launch ResampleScalarVectorDWIVolume "
            f"{preprocessed_seg} {output_seg} -i nn -f {transform_outfile} -R {main_SS_file} -z {x},{y},{z}"
        )
        if verbose:
            print(command)
        try:
            run(command.split(" "), capture_output=not verbose).check_returncode()
            sitk_check(output_seg)

        except Exception as error:
            return row, error

    return row, None


def preprocess_study(
    study_df: pd.DataFrame,
    preprocessed_dir: Path | str,
    pipeline_key: str,
    registration_key: str = "T1Post",
    registration_target: str | None = None,
    orientation: str = "RAI",
    spacing: str = "1,1,1",
    skullstrip: bool = True,
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
        'StudyInstanceUID', 'NormalizedSeriesDescription', and 'SeriesType'.
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
            sitk_check(preprocessed_file)
        else:
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy(input_file, preprocessed_file)
            if os.path.exists(preprocessed_file):
                rows[i][pipeline_key] = str(preprocessed_file)
            else:
                error = f"Could not create {preprocessed_file}"
                print(error)
                e = open(f"{preprocessed_dir}/errors.txt", "a")
                e.write(f"{error}\n")
                return study_df

        if "seg" in rows[i]:
            input_file = rows[i]["seg"]
            preprocessed_seg = output_dir / os.path.basename(input_file)
            shutil.copy(input_file, preprocessed_seg)

            if os.path.exists(preprocessed_seg):
                rows[i][f"{pipeline_key}_seg"] = str(preprocessed_seg)
                sitk_check(preprocessed_seg)
            else:
                os.makedirs(output_dir, exist_ok=True)
                shutil.copy(input_file, preprocessed_seg)
                if os.path.exists(preprocessed_file):
                    rows[i][f"{pipeline_key}_seg"] = str(preprocessed_file)
                else:
                    error = f"Could not create {preprocessed_seg}"
                    print(error)
                    e = open(f"{preprocessed_dir}/errors.txt", "a")
                    e.write(f"{error}\n")
                    return study_df

    ### Orientation
    for i in range(n):
        preprocessed_file = rows[i][pipeline_key]

        if debug:
            output_file = preprocessed_file.replace(".nii.gz", "_orientation.nii.gz")
            rows[i][pipeline_key] = output_file

        else:
            output_file = preprocessed_file

        command = (
            f"Slicer --launch OrientScalarVolume "
            f"{preprocessed_file} {output_file} -o {orientation}"
        )
        if verbose:
            print(command)
        try:
            run(command.split(" "), capture_output=not verbose).check_returncode()
            sitk_check(output_file)

        except Exception as error:
            print(error)
            e = open(f"{preprocessed_dir}/errors.txt", "a")
            e.write(f"{error}\n")
            return study_df

        if "seg" in rows[i]:
            preprocessed_seg = rows[i][f"{pipeline_key}_seg"]

            if debug:
                output_seg = preprocessed_seg.replace(".nii.gz", "_orientation.nii.gz")
                rows[i][f"{pipeline_key}_seg"] = output_seg

            else:
                output_seg = preprocessed_seg

            command = (
                f"Slicer --launch OrientScalarVolume "
                f"{preprocessed_seg} {output_seg} -o {orientation}"
            )
            if verbose:
                print(command)
            try:
                run(command.split(" "), capture_output=not verbose).check_returncode()
                sitk_check(output_seg)

            except Exception as error:
                print(error)
                e = open(f"{preprocessed_dir}/errors.txt", "a")
                e.write(f"{error}\n")
                return study_df

    ### Spacing
    for i in range(n):
        preprocessed_file = rows[i][pipeline_key]

        if debug:
            output_file = preprocessed_file.replace(".nii.gz", "_spacing.nii.gz")
            rows[i][pipeline_key] = output_file

        else:
            output_file = preprocessed_file

        command = (
            f"Slicer --launch ResampleScalarVolume "
            f"{preprocessed_file} {output_file} -i bspline -s {spacing}"
        )
        if verbose:
            print(command)
        try:
            run(command.split(" "), capture_output=not verbose).check_returncode()
            sitk_check(output_file)

        except Exception as error:
            print(error)
            e = open(f"{preprocessed_dir}/errors.txt", "a")
            e.write(f"{error}\n")
            return study_df

        if "seg" in rows[i]:
            preprocessed_seg = rows[i][f"{pipeline_key}_seg"]

            if debug:
                output_seg = preprocessed_seg.replace(".nii.gz", "_spacing.nii.gz")
                rows[i][f"{pipeline_key}_seg"] = output_seg

            else:
                output_seg = preprocessed_seg

            command = (
                f"Slicer --launch ResampleScalarVolume "
                f"{preprocessed_seg} {output_seg} -i nearestNeighbor -s {spacing}"
            )
            if verbose:
                print(command)
            try:
                run(command.split(" "), capture_output=not verbose).check_returncode()
                sitk_check(output_seg)

            except Exception as error:
                print(error)
                e = open(f"{preprocessed_dir}/errors.txt", "a")
                e.write(f"{error}\n")
                return study_df

    ### Loose Skullstrip
    for i in range(n):
        preprocessed_file = rows[i][pipeline_key]
        SS_file = preprocessed_file.replace(".nii.gz", "_SS.nii.gz")
        SS_mask = preprocessed_file.replace(".nii.gz", "_SS_mask.nii.gz")

        command = f"mri_synthstrip -i {preprocessed_file} -o {SS_file} -m {SS_mask}"
        if verbose:
            print(command)
        try:
            run(command.split(" "), capture_output=not verbose).check_returncode()
            sitk_check(SS_file)
            sitk_check(SS_mask)

        except Exception as error:
            print(error)
            e = open(f"{preprocessed_dir}/errors.txt", "a")
            e.write(f"{error}\n")
            return study_df

    if debug:
        if registration_target is None:
            main_SS_file = rows[0][pipeline_key].replace(".nii.gz", "_SS.nii.gz")
            main_SS_mask_file = rows[0][pipeline_key].replace(
                ".nii.gz", "_SS_mask.nii.gz"
            )
            main_SS_mask_array = np.round(
                as_closest_canonical(load(main_SS_mask_file)).get_fdata()
            )
        else:
            main_SS_file = registration_target.replace(
                ".nii.gz", "_orientation_spacing_SS.nii.gz"
            )
            main_SS_mask_file = registration_target.replace(
                ".nii.gz", "_orientation_spacing_SS_mask.nii.gz"
            )
            main_SS_mask_array = np.round(
                as_closest_canonical(load(main_SS_mask_file)).get_fdata()
            )

    else:
        if registration_target is None:
            main_SS_file = rows[0][pipeline_key].replace(".nii.gz", "_SS.nii.gz")
            main_SS_mask_file = rows[0][pipeline_key].replace(
                ".nii.gz", "_SS_mask.nii.gz"
            )
            main_SS_mask_array = np.round(
                as_closest_canonical(load(main_SS_mask_file)).get_fdata()
            )
        else:
            main_SS_file = registration_target.replace(".nii.gz", "_SS.nii.gz")
            main_SS_mask_file = registration_target.replace(
                ".nii.gz", "_SS_mask.nii.gz"
            )
            main_SS_mask_array = np.round(
                as_closest_canonical(load(main_SS_mask_file)).get_fdata()
            )

    ### Register based on loose skullstrip
    fixed_image_path = main_SS_file
    if registration_target is None:
        if n > 1:
            for i in range(1, n):
                registered_row, error = registration(
                    rows[i],
                    pipeline_key,
                    fixed_image_path,
                    main_SS_file,
                    main_SS_mask_file,
                    main_SS_mask_array,
                    verbose,
                    debug,
                )

                if error is not None:
                    print(error)
                    e = open(f"{preprocessed_dir}/errors.txt", "a")
                    e.write(f"{error}\n")
                    return study_df

                rows[i] = registered_row

    else:
        for i in range(n):
            registered_row, error = registration(
                rows[i],
                pipeline_key,
                fixed_image_path,
                main_SS_file,
                main_SS_mask_file,
                main_SS_mask_array,
                verbose,
                debug,
            )

            if error is not None:
                print(error)
                e = open(f"{preprocessed_dir}/errors.txt", "a")
                e.write(f"{error}\n")
                return study_df

            rows[i] = registered_row

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

            nifti = as_closest_canonical(load(preprocessed_file))
            array = nifti.get_fdata()

            array = array * main_SS_mask_array

            output_nifti = Nifti1Image(array, affine=nifti.affine, header=nifti.header)
            save(output_nifti, output_file)
            sitk_check(output_file)

            if "seg" in rows[i]:
                preprocessed_seg = rows[i][f"{pipeline_key}_seg"]

                if debug:
                    output_seg = preprocessed_seg.replace(
                        ".nii.gz", "_skullstripped.nii.gz"
                    )
                    rows[i][f"{pipeline_key}_seg"] = output_seg

                else:
                    output_seg = preprocessed_seg

                nifti = as_closest_canonical(load(preprocessed_seg))
                array = nifti.get_fdata()

                array = array * main_SS_mask_array

                output_nifti = Nifti1Image(
                    array, affine=nifti.affine, header=nifti.header
                )
                save(output_nifti, output_seg)
                sitk_check(output_seg)

    ### Bias correction
    for i in range(n):
        preprocessed_file = rows[i][pipeline_key]

        if debug:
            output_file = preprocessed_file.replace(".nii.gz", "_N4.nii.gz")
            rows[i][pipeline_key] = output_file

        else:
            output_file = preprocessed_file

        n4 = N4BiasFieldCorrection()
        n4.inputs.input_image = preprocessed_file
        n4.inputs.n_iterations = [20, 20, 10, 5]
        n4.inputs.output_image = output_file
        n4.run()
        sitk_check(output_file)

    ### Normalization
    for i in range(n):
        preprocessed_file = rows[i][pipeline_key]

        if debug:
            output_file = preprocessed_file.replace(".nii.gz", "_norm.nii.gz")
            rows[i][pipeline_key] = output_file

        else:
            output_file = preprocessed_file

        nifti = as_closest_canonical(load(preprocessed_file))
        array = nifti.get_fdata()
        masked_input_array = np.ma.masked_where(main_SS_mask_array == 0, array)
        mean = np.ma.mean(masked_input_array)
        std = np.ma.std(masked_input_array)
        array = (array - mean) / (std + 1e-6)

        output_nifti = as_closest_canonical(
            Nifti1Image(array, affine=nifti.affine, header=nifti.header)
        )
        save(output_nifti, output_file)
        sitk_check(output_file)

    ### copy metadata
    preprocessing_args = {
        "preprocessed_dir": str(preprocessed_dir),
        "pipeline_key": pipeline_key,
        "registration_target": registration_target,
        "orientation": orientation,
        "spacing": spacing,
        "skullstrip": skullstrip,
    }

    for row in rows:
        copy_metadata(row, preprocessing_args)

    preprocessed_df = pd.DataFrame(rows)
    out_df = pd.merge(study_df, preprocessed_df, "outer")

    return out_df


def preprocess_patient(
    patient_df: pd.DataFrame,
    preprocessed_dir: Path | str,
    pipeline_key: str = "preprocessed",
    registration_key: str = "T1Post",
    longitudinal_registration: bool = False,
    orientation: str = "RAI",
    spacing: str = "1,1,1",
    skullstrip: bool = True,
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
        'StudyInstanceUID', 'NormalizedSeriesDescription', and 'SeriesType'.
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
    registration_target: str | None
        The location of the file that will be used as the fixed image for the purposes of registration.
    orientation: str
        The orientation standard that you wish to set for preprocessed data. Defaults to 'RAI'."
    spacing: str
        A comma delimited list indicating the desired spacing of preprocessed data. Measurements
        are in mm. Defaults to '1,1,1'.
    skullstrip: bool
        Whether to apply skullstripping to preprocessed data. Skullstripping will be applied by default.
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

    preprocessed_dfs.append(
        preprocess_study(
            study_df=study_df,
            preprocessed_dir=preprocessed_dir,
            pipeline_key=pipeline_key,
            registration_key=registration_key,
            registration_target=None,
            orientation=orientation,
            spacing=spacing,
            skullstrip=skullstrip,
            verbose=verbose,
            source_software=False,
            check_columns=False,
            debug=debug,
        )
    )

    if len(study_uids) > 1:
        if longitudinal_registration:
            if debug:
                output_dir = os.path.dirname(preprocessed_dfs[0][pipeline_key][0])
                base_name = os.path.basename(preprocessed_dfs[0]["nifti"][0])
                registration_target = f"{output_dir}/{base_name}"

            else:
                registration_target = preprocessed_dfs[0][pipeline_key][0]

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
                        orientation=orientation,
                        spacing=spacing,
                        skullstrip=skullstrip,
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
                        orientation=orientation,
                        spacing=spacing,
                        skullstrip=skullstrip,
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
            + list(patient_dir.glob("**/*SS_mask.nii.gz"))
            + list(patient_dir.glob("**/*.tfm"))
            + list(patient_dir.glob("**/*.h5"))
        )

        print("......Clearing unnecessary files......")
        for file in extra_files:
            os.remove(file)

    print(f"Finished preprocessing {anon_patientID}:")
    print(out_df)
    return out_df


def preprocess_patient_star(args):
    """
    Helper function intended for use only in 'preprocess_from_csv'. Provides an imap
    compatible version of 'preprocess_patient'.
    """

    return preprocess_patient(*args)


def preprocess_from_csv(
    csv: Path | str,
    preprocessed_dir: Path | str,
    pipeline_key: str = "preprocessed",
    registration_key: str = "T1Post",
    longitudinal_registration: bool = False,
    orientation: str = "RAI",
    spacing: str = "1,1,1",
    skullstrip: bool = True,
    cpus: int = 0,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Preprocess all of the studies for a patient in a DataFrame.

    Parameters
    __________
    csv: Path | str
        The path to a CSV containing an entire dataset. It must contain the following columns:  'nifti',
        'Anon_PatientID', 'Anon_StudyID', 'StudyInstanceUID', 'NormalizedSeriesDescription', and 'SeriesType'.
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
    registration_target: str | None
        The location of the file that will be used as the fixed image for the purposes of registration.
    orientation: str
        The orientation standard that you wish to set for preprocessed data. Defaults to 'RAI'."
    spacing: str
        A comma delimited list indicating the desired spacing of preprocessed data. Measurements
        are in mm. Defaults to '1,1,1'.
    skullstrip: bool
        Whether to apply skullstripping to preprocessed data. Skullstripping will be applied by default.
    verbose: bool
        Whether to print additional information related like commands and their arguments are printed.
    source_software: bool
        Whether to call `source_external_software` to add software required for preprocessing. Defaults
        to True.
    check_columns: bool
        Whether to check `study_df` for required columns. Defaults to True.

    Returns
    _______
    pd.DataFrame:
        A Dataframe with added column f'{pipeline_key}' and optionally f'{pipeline_key}_seg' to indicate
        the locations of the preprocessing outputs. This function will also overwrite the input CSV with
        this DataFrame.
    """

    source_external_software()

    df = pd.read_csv(csv)

    if pipeline_key in df.keys():
        df = df.drop(columns=pipeline_key)

    required_columns = [
        "nifti",
        "Anon_PatientID",
        "Anon_StudyID",
        "StudyInstanceUID",
        "NormalizedSeriesDescription",
        "SeriesType",
    ]
    optional_columns = ["seg"]

    check_required_columns(df, required_columns, optional_columns)

    preprocessed_dir = Path(preprocessed_dir)

    filtered_df = df.copy().dropna(subset="nifti")
    patients = filtered_df["Anon_PatientID"].unique()

    if cpus == 0:
        outputs = [
            preprocess_patient(
                filtered_df[filtered_df["Anon_PatientID"] == patient].copy(),
                preprocessed_dir,
                pipeline_key,
                registration_key,
                longitudinal_registration,
                orientation,
                spacing,
                skullstrip,
                verbose,
                False,
                False,
            )
            for patient in tqdm(patients, desc="Preprocessing patients")
        ]

    else:
        inputs = [
            [
                filtered_df[filtered_df["Anon_PatientID"] == patient].copy(),
                preprocessed_dir,
                pipeline_key,
                registration_key,
                longitudinal_registration,
                orientation,
                spacing,
                skullstrip,
                verbose,
                False,
                False,
            ]
            for patient in patients
        ]

        with multiprocessing.Pool(cpus) as pool:
            outputs = list(
                tqdm(
                    pool.imap(preprocess_patient_star, inputs),
                    total=len(patients),
                    desc="Preprocessing patients",
                )
            )

    preprocessed_df = pd.concat(outputs)
    df = pd.merge(df, preprocessed_df, how="outer")
    df = df.sort_values(["Anon_PatientID", "Anon_StudyID"]).reset_index(drop=True)
    df.to_csv(csv, index=False)
    return df


def debug_from_csv(
    csv: Path | str,
    preprocessed_dir: Path | str,
    patients: Sequence[str] | None = None,
    pipeline_key: str = "debug",
    registration_key: str = "T1Post",
    longitudinal_registration: bool = False,
    orientation: str = "RAI",
    spacing: str = "1,1,1",
    skullstrip: bool = True,
    cpus: int = 0,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Preprocess all of the studies for a patient in a DataFrame.

    Parameters
    __________
    csv: Path | str
        The path to a CSV containing an entire dataset. It must contain the following columns:  'nifti',
        'Anon_PatientID', 'Anon_StudyID', 'StudyInstanceUID', 'NormalizedSeriesDescription', and 'SeriesType'.
    preprocessed_dir: Path
        The directory that will contain the preprocessed NIfTI files.
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
    registration_target: str | None
        The location of the file that will be used as the fixed image for the purposes of registration.
    orientation: str
        The orientation standard that you wish to set for preprocessed data. Defaults to 'RAI'."
    spacing: str
        A comma delimited list indicating the desired spacing of preprocessed data. Measurements
        are in mm. Defaults to '1,1,1'.
    skullstrip: bool
        Whether to apply skullstripping to preprocessed data. Skullstripping will be applied by default.
    verbose: bool
        Whether to print additional information related like commands and their arguments are printed.
    source_software: bool
        Whether to call `source_external_software` to add software required for preprocessing. Defaults
        to True.
    check_columns: bool
        Whether to check `study_df` for required columns. Defaults to True.

    Returns
    _______
    pd.DataFrame:
        A Dataframe with added column f'{pipeline_key}' and optionally f'{pipeline_key}_seg' to indicate
        the locations of the preprocessing outputs. This function will also overwrite the input CSV with
        this DataFrame.
    """

    source_external_software()

    df = pd.read_csv(csv)

    if pipeline_key in df.keys():
        df = df.drop(columns=pipeline_key)

    required_columns = [
        "nifti",
        "Anon_PatientID",
        "Anon_StudyID",
        "StudyInstanceUID",
        "NormalizedSeriesDescription",
        "SeriesType",
    ]
    optional_columns = ["seg"]

    check_required_columns(df, required_columns, optional_columns)

    preprocessed_dir = Path(preprocessed_dir)

    filtered_df = df.copy().dropna(subset="nifti")

    if patients is None:
        patients = filtered_df["Anon_PatientID"].unique()

    if cpus == 0:
        outputs = [
            preprocess_patient(
                filtered_df[filtered_df["Anon_PatientID"] == patient].copy(),
                preprocessed_dir,
                pipeline_key,
                registration_key,
                longitudinal_registration,
                orientation,
                spacing,
                skullstrip,
                verbose,
                False,
                False,
                True,
            )
            for patient in tqdm(patients, desc="Preprocessing patients")
        ]

    else:
        inputs = [
            [
                filtered_df[filtered_df["Anon_PatientID"] == patient].copy(),
                preprocessed_dir,
                pipeline_key,
                registration_key,
                longitudinal_registration,
                orientation,
                spacing,
                skullstrip,
                verbose,
                False,
                False,
                True,
            ]
            for patient in patients
        ]

        with multiprocessing.Pool(cpus) as pool:
            outputs = list(
                tqdm(
                    pool.imap(preprocess_patient_star, inputs),
                    total=len(patients),
                    desc="Preprocessing patients",
                )
            )

    preprocessed_df = pd.concat(outputs)
    preprocessed_df = preprocessed_df.sort_values(
        ["Anon_PatientID", "Anon_StudyID"]
    ).reset_index(drop=True)
    preprocessed_df.to_csv(preprocessed_dir / "debug.csv", index=False)
    return preprocessed_df
