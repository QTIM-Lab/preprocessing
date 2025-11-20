"""
The `liver_ct` module defines the tools for processing CT data
with a pipeline designed specifically for liver data.

Public Functions
----------------
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
import datetime

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
from preprocessing.utils import (
    check_required_columns,
    cpu_adjust,
    update_errorfile
)

from typing import Sequence, List, Literal, Dict, Any, Tuple
from scipy.ndimage import binary_fill_holes, generate_binary_structure
from cc3d import connected_components
from concurrent.futures import ProcessPoolExecutor, as_completed


def copy_metadata(row: Dict[str, Any], preprocessing_args: Dict[str, Any]) -> None:
    """
    Copy the metadata file paired with the original NIfTI file (and optionally the
    corresponding segmentation) and add the preprocessing arguments into a new metafile
    to be paired with the preprocessing outputs.

    Parameters
    ----------
    row: dict
        A row of a DataFrame represented as a dictionary. It is expected to have a 'Nifti'
        key and optionally 'Seg'.

    preprocessing_args: dict
        A dictionary containing the arguments originally provided to 'preprocess_study' or
        'preprocess_from_csv'.

    Returns
    -------
    None
        A metadata json is saved out to be paired with the preprocessed outputs.

    """
    original_metafile = row["Nifti"].replace(".nii.gz", ".json")
    if Path(original_metafile).exists():
        try:
            with open(original_metafile, "r") as json_file:
                data = json.load(json_file)
        except Exception:
            data = os.path.abspath(original_metafile)
        meta_dict = {
            "source_file": row["Nifti"],
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
            "source_file": row["Nifti"],
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

    if "Seg" in row and not pd.isna(row["Seg"]):
        original_metafile = row["Seg"].replace(".nii.gz", ".json")
        if Path(original_metafile).exists():
            try:
                with open(original_metafile, "r") as json_file:
                    data = json.load(json_file)
            except Exception:
                data = os.path.abspath(original_metafile)
            meta_dict = {
                "source_file": row["Nifti"],
                "original_metafile": data,
                "preprocessing_args": preprocessing_args,
            }
            preprocessed_metafile = row[
                f"{preprocessing_args['pipeline_key']}Seg"
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
                "source_file": row["Nifti"],
                "original_metafile": None,
                "preprocessing_args": preprocessing_args,
            }
            preprocessed_metafile = row[
                f"{preprocessing_args['pipeline_key']}Seg"
            ].replace(".nii.gz", ".json")
            with open(preprocessed_metafile, "w") as json_file:
                json.dump(
                    meta_dict,
                    json_file,
                    sort_keys=True,
                    indent=2,
                    separators=(",", ": "),
                )


def fill_foreground_mask(initial_foreground: np.ndarray) -> np.ndarray:
    """
    Fill the initial foreground mask so that it will include the entire foreground.

    Parameters
    ----------
    initial_foreground: np.ndarray
        The initial foreground mask that represents the border of the foreground but is not filled.

    Returns
    -------
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
    pipeline_key: str = "Preprocessed",
    orientation: str = "RAS",
    spacing: Sequence[float | int] = [1, 1, 1],
    binarize_seg: bool = False,
    verbose: bool = False,
    check_columns: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Preprocess a single study from a DataFrame.

    Parameters
    ----------
    study_df: pd.DataFrame
        A DataFrame containing NIfTI location and information required for the output file names
        for a single study. It must contain the columns: 'Nifti', 'AnonPatientID', 'AnonStudyID',
        'StudyInstanceUID', 'SeriesInstanceUID', 'NormalizedSeriesDescription', and 'SeriesType'.

    preprocessed_dir: Path
        The directory that will contain the preprocessed NIfTI files.

    pipeline_key: str
        The key that will be added to the DataFrame to indicate the new locations of preprocessed files.
        Defaults to 'Preprocessed'.

    orientation: str
        The orientation standard that you wish to set for preprocessed data. Defaults to 'RAI'."

    spacing: Sequence[float | int]
        A sequence of floats or ints indicating the desired spacing of preprocessed data. Measurements
        are in mm. Defaults to [1, 1, 1].

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
    -------
    pd.DataFrame:
        A Dataframe with added column f'{pipeline_key}' and optionally f'{pipeline_key}Seg' to indicate
        the locations of the preprocessing outputs.
    """
    if check_columns:
        required_columns = [
            "Nifti",
            "AnonPatientID",
            "AnonStudyID",
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "NormalizedSeriesDescription",
        ]
        optional_columns = ["Seg", "SeriesType"]

        check_required_columns(study_df, required_columns, optional_columns)

    preprocessed_dir = Path(preprocessed_dir).resolve()

    filtered_df = (
        study_df.copy()
        .dropna(subset="NormalizedSeriesDescription")
    #     .sort_values(
    #         ["NormalizedSeriesDescription"],
    #         key=lambda x: (x != registration_key).astype(int),
    #     )
    )
    if filtered_df.empty:
        return study_df

    anon_patientID = filtered_df.loc[filtered_df.index[0], "AnonPatientID"]
    anon_studyID = filtered_df.loc[filtered_df.index[0], "AnonStudyID"]

    rows = filtered_df.to_dict("records")
    n = len(rows)

    sitk_im_cache = {}

    # must enforce one normalizedseries description per study
    ### copy files to new location
    for i in range(n):
        output_dir = (
            preprocessed_dir / anon_patientID / anon_studyID / rows[i].get("SeriesType", "anat")
        )
        os.makedirs(output_dir, exist_ok=True)

        input_file = rows[i]["Nifti"]
        preprocessed_file = output_dir / os.path.basename(input_file)

        if ".gz" not in str(preprocessed_file):
            preprocessed_file = Path(str(preprocessed_file) + ".gz")

        rows[i][pipeline_key] = str(preprocessed_file)

        sitk_im_cache[str(preprocessed_file)] = ReadImage(input_file, imageIO="NiftiImageIO")

        if "Seg" in rows[i] and not pd.isna(rows[i]["Seg"]):
            input_seg = rows[i]["Seg"]
            preprocessed_seg = output_dir / os.path.basename(input_seg)

            if ".gz" not in str(preprocessed_seg):
                preprocessed_seg = Path(str(preprocessed_seg) + ".gz")

            rows[i][f"{pipeline_key}Seg"] = str(preprocessed_seg)

            sitk_im_cache[str(preprocessed_seg)] = ReadImage(input_seg, imageIO="NiftiImageIO")

    ### Optionally enforce binary segmentations
    if binarize_seg:
        for i in range(n):
            if "Seg" in rows[i] and not pd.isna(rows[i]["Seg"]):
                preprocessed_seg = rows[i][f"{pipeline_key}Seg"]

                if debug:
                    output_seg = preprocessed_seg.replace(".nii", "_binary.nii")
                    rows[i][f"{pipeline_key}Seg"] = output_seg

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
            output_file = preprocessed_file.replace(".nii", f"_{orientation}.nii")
            rows[i][pipeline_key] = output_file

        else:
            output_file = preprocessed_file

        nifti = sitk_im_cache[preprocessed_file]
        output_nifti = orienter.Execute(nifti)

        sitk_im_cache[output_file] = output_nifti

        if verbose:
            print(f"{preprocessed_file} set to {orientation} orientation")

        if "Seg" in rows[i] and not pd.isna(rows[i]["Seg"]):
            preprocessed_seg = rows[i][f"{pipeline_key}Seg"]

            if debug:
                output_seg = preprocessed_seg.replace(".nii", f"_{orientation}.nii")
                rows[i][f"{pipeline_key}Seg"] = output_seg

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
            output_file = preprocessed_file.replace(".nii", "_spacing.nii")
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

        if "Seg" in rows[i] and not pd.isna(rows[i]["Seg"]):
            preprocessed_seg = rows[i][f"{pipeline_key}Seg"]

            if debug:
                output_seg = preprocessed_seg.replace(".nii", "_spacing.nii")
                rows[i][f"{pipeline_key}Seg"] = output_seg

            else:
                output_seg = preprocessed_seg

            nifti = sitk_im_cache[preprocessed_seg]
            output_nifti = Resample(
                nifti,
                sitk_im_cache[preprocessed_file],
                interpolator=sitkNearestNeighbor,
            )

            sitk_im_cache[output_seg] = output_nifti

            if verbose:
                print(f"{preprocessed_seg} resampled to {spacing} spacing")


    ### foreground
    foreground_file = rows[0][pipeline_key].replace(
        ".nii", "_foreground_mask.nii"
    )
    nifti = sitk_im_cache[rows[0][pipeline_key]]

    # threshold_filter = OtsuThresholdImageFilter()
    # threshold_filter.Execute(nifti)
    # threshold = threshold_filter.GetThreshold()
    #
    # foreground_array = (GetArrayFromImage(nifti) >= threshold).astype(int)

    background_array = (GetArrayFromImage(nifti) < -850)
    background_cc = connected_components(background_array)
    ccs, counts = np.unique(background_cc, return_counts=True)

    sorted_ccs = sorted(
        [(cc, count) if cc != 0 else (0, 0) for cc, count in zip(ccs, counts)],
        key=lambda x: x[1],
    )

    largest_cc = sorted_ccs[-1][0]

    background_array = (background_cc == largest_cc).astype(int)

    foreground_array = 1 - background_array

    foreground_array = fill_foreground_mask(foreground_array)

    foreground = GetImageFromArray(foreground_array)
    foreground.CopyInformation(nifti)

    sitk_im_cache[foreground_file] = foreground

    ### Normalization + skullstripping
    for i in range(n):
        preprocessed_file = rows[i][pipeline_key]

        if debug:
            output_file = preprocessed_file.replace(".nii", "_norm.nii")
            rows[i][pipeline_key] = output_file

        else:
            output_file = preprocessed_file

        nifti = sitk_im_cache[preprocessed_file]
        array = GetArrayFromImage(nifti)

        array = np.clip(array, -160, 240)

        output_nifti = GetImageFromArray(array)
        output_nifti.CopyInformation(nifti)

        sitk_im_cache[output_file] = output_nifti

        if verbose:
            print(f"{preprocessed_file} intensity normalized")

        array = array * foreground_array

        preprocessed_file = rows[i][pipeline_key]

        if debug:
            output_file = preprocessed_file.replace(".nii", "_0background.nii")
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
        if "integer" in v.GetPixelIDTypeAsString():
            WriteImage(Cast(v, sitkUInt8), k, compressionLevel=6)
        else:
            WriteImage(Cast(v, sitkFloat32), k, compressionLevel=6)

    preprocessed_df = pd.DataFrame(rows)
    out_df = pd.merge(study_df, preprocessed_df, "outer")

    return out_df


def preprocess_patient(
    patient_df: pd.DataFrame,
    preprocessed_dir: Path | str,
    pipeline_key: str = "Preprocessed",
    orientation: str = "RAS",
    spacing: Sequence[float | int] = [1, 1, 1],
    binarize_seg: bool = False,
    verbose: bool = False,
    check_columns: bool = True,
    debug: bool = False,
):
    """
    Preprocess all of the studies for a patient in a DataFrame.

    Parameters
    ----------
    patient_df: pd.DataFrame
        A DataFrame containing nifti location and information required for the output file names
        for a single patient. It must contain the columns: 'Nifti', 'AnonPatientID', 'AnonStudyID',
        'StudyInstanceUID', 'SeriesInstanceUID', 'NormalizedSeriesDescription', and 'SeriesType'.

    preprocessed_dir: Path
        The directory that will contain the preprocessed NIfTI files.

    orientation: str
        The orientation standard that you wish to set for preprocessed data. Defaults to 'RAS'.

    spacing: Sequence[float | int]
        A sequence of floats or ints indicating the desired spacing of preprocessed data. Measurements
        are in mm. Defaults to [1, 1, 1].

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
    -------
    pd.DataFrame:
        A Dataframe with added column f'{pipeline_key}' and optionally f'{pipeline_key}Seg' to indicate
        the locations of the preprocessing outputs.
    """

    if check_columns:
        required_columns = [
            "Nifti",
            "AnonPatientID",
            "AnonStudyID",
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "NormalizedSeriesDescription",
        ]
        optional_columns = ["Seg", "SeriesType"]

        check_required_columns(patient_df, required_columns, optional_columns)

    if patient_df.shape[0] == 0:
        return patient_df

    preprocessed_dir = Path(preprocessed_dir).resolve()

    study_uids = patient_df["StudyInstanceUID"].unique()

    preprocessed_dfs = []

    for study_uid in study_uids:
        study_df = patient_df[
            patient_df["StudyInstanceUID"] == study_uid
        ].copy()

        preprocessed_dfs.append(
            preprocess_study(
                study_df=study_df,
                preprocessed_dir=preprocessed_dir,
                pipeline_key=pipeline_key,
                orientation=orientation,
                spacing=spacing,
                binarize_seg=binarize_seg,
                verbose=verbose,
                check_columns=False,
                debug=debug,
            )
        )

    # clear extra files
    anon_patientID = patient_df.loc[patient_df.index[0], "AnonPatientID"]
    patient_dir = preprocessed_dir / anon_patientID

    out_df = pd.concat(preprocessed_dfs, ignore_index=True)

    if not debug:
        extra_files = (
            list(patient_dir.glob("**/*SS.nii*"))
            + list(patient_dir.glob("**/*mask.nii*"))
            + list(patient_dir.glob("**/*longreg.nii*"))
            # + list(patient_dir.glob("**/*.mgz"))
            # + list(patient_dir.glob("**/*.m3z"))
            # + list(patient_dir.glob("**/*.txt"))
        )

        # print("......Clearing unnecessary files......")
        for file in extra_files:
            os.remove(file)

    # print(f"Finished preprocessing {anon_patientID}:")
    # print(out_df)
    return out_df


def preprocess_from_csv(
    csv: Path | str,
    preprocessed_dir: Path | str,
    patients: Sequence[str] | None = None,
    pipeline_key: str = "Preprocessed",
    orientation: str = "RAS",
    spacing: Sequence[float | int] = [1, 1, 1],
    binarize_seg: bool = False,
    cpus: int = 1,
    verbose: bool = False,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Preprocess all of the studies within a dataset.

    Parameters
    ----------
    csv: Path | str
        The path to a CSV containing an entire dataset. It must contain the following columns:  'Nifti',
        'AnonPatientID', 'AnonStudyID', 'StudyInstanceUID', 'SeriesInstanceUID', 'NormalizedSeriesDescription',
        and 'SeriesType'.

    preprocessed_dir: Path
        The directory that will contain the preprocessed NIfTI files.

    patients: Sequece[str] | None
        A sequence of patients to select from the 'AnonPatientID' column of the CSV. If 'None' is provided,
        all patients will be preprocessed.

    pipeline_key: str
        The key that will be added to the DataFrame to indicate the new locations of preprocessed files.
        Defaults to 'Preprocessed'.

    orientation: str
        The orientation standard that you wish to set for preprocessed data. Defaults to 'RAS'.

    spacing: Sequence[float | int]
        A sequence of floats or ints indicating the desired spacing of preprocessed data. Measurements
        are in mm. Defaults to [1, 1, 1].

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
    -------
    pd.DataFrame:
        A Dataframe with added column f'{pipeline_key}' and optionally f'{pipeline_key}Seg' to indicate
        the locations of the preprocessing outputs. This function will also overwrite the input CSV with
        this DataFrame.
    """
    df = pd.read_csv(csv, dtype=str)

    preprocessed_dir = Path(preprocessed_dir).resolve()

    if debug:
        csv = preprocessed_dir / "debug.csv"
        pipeline_key = "debug"

    elif all(var in os.environ for var in ["SLURM_ARRAY_TASK_ID", "SLURM_ARRAY_OUTPUTS"]):
        csv = Path(os.environ['SLURM_ARRAY_OUTPUTS']).resolve() / f"{os.environ['SLURM_ARRAY_TASK_ID']}.csv"

    if pipeline_key in df.keys():
        df = df.drop(columns=pipeline_key)
        if f"{pipeline_key}Seg" in df.keys():
            df = df.drop(columns=f"{pipeline_key}Seg")

    df.to_csv(csv, index=False)

    required_columns = [
        "Nifti",
        "AnonPatientID",
        "AnonStudyID",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "NormalizedSeriesDescription",
    ]
    optional_columns = ["Seg", "SeriesType"]

    check_required_columns(df, required_columns, optional_columns)

    preprocessed_dir = Path(preprocessed_dir).resolve()
    errorfile = preprocessed_dir / f"{str(datetime.datetime.now()).replace(' ', '_')}.txt"

    df = df.drop_duplicates(subset="SeriesInstanceUID").reset_index(drop=True)

    filtered_df = df.copy().dropna(subset="Nifti")

    if patients is None:
        patients = list(filtered_df["AnonPatientID"].unique())

    if "SLURM_ARRAY_TASK_ID" in os.environ:
        patients = [patients[int(os.environ["SLURM_ARRAY_TASK_ID"])]]

    kwargs_list = [
        {
            "patient_df": filtered_df[filtered_df["AnonPatientID"] == patient].copy(),
            "preprocessed_dir": preprocessed_dir,
            "pipeline_key": pipeline_key,
            "orientation": orientation,
            "spacing": spacing,
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
        futures = {
            executor.submit(preprocess_patient, **kwargs): kwargs
            for kwargs in kwargs_list
        }

        for future in as_completed(futures.keys()):
            try:
                preprocessed_df = future.result()

            except Exception as error:
                update_errorfile(
                    func_name="preprocessing.brain.preprocess_patient",
                    kwargs=futures[future],
                    errorfile=errorfile,
                    error=error
                )

                pbar.update(1)
                continue

            df = (
                pd.read_csv(csv, dtype=str)
                .drop_duplicates(subset="SeriesInstanceUID")
                .reset_index(drop=True)
            )
            df = pd.merge(df, preprocessed_df, how="outer")
            df = (
                df.drop_duplicates(subset="SeriesInstanceUID")
                .sort_values(["AnonPatientID", "AnonStudyID"])
                .reset_index(drop=True)
            )
            df.to_csv(csv, index=False)


            pbar.update(1)

    df = (
        pd.read_csv(csv, dtype=str)
        .drop_duplicates(subset="SeriesInstanceUID")
        .sort_values(["AnonPatientID", "AnonStudyID"])
        .reset_index(drop=True)
    )
    df.to_csv(csv, index=False)
    return df


__all__ = [
    "preprocess_study",
    "preprocess_patient",
    "preprocess_from_csv",
]
