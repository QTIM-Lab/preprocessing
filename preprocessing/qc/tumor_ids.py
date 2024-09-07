"""
The `tumor_ids` module defines the tools for longitudinal tracking of
individual tumors. Each connected component for a given label within a segmentation mask
is assigned unique ID that will remain consistent across all scans belonging
to the same patient patient. This code assumes that the longitudinal or atlas registration
was used when preprocessing the data.

Public Functions
----------------
track_patient_tumors
    Assign tumor IDs in all of the segmentations masks for a single patient.

track_tumors_csv
    Assign tumor IDs in all of the segmentations masks for every patient within a dataset.
"""
import pandas as pd
import numpy as np
import os
import datetime

from preprocessing.utils import check_required_columns, update_errorfile
from SimpleITK import ReadImage, WriteImage, GetArrayFromImage, GetImageFromArray
from pathlib import Path
from typing import Dict, Sequence
from cc3d import connected_components
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def assign_tumor_ids(
    cc_arrays: Dict[str, Dict[str, np.array]]
) -> Dict[str, Dict[str, np.array]]:
    """
    Assign unique IDs to the provided connected component arrays.

    Parameters
    ----------
    cc_arrays: Dict[str, Dict[str, np.array]]
        A nested dictionary mapping the current label and corresponding file path to an
        array containing connected components.


    Returns
    -------
    cc_arrays: Dict[str, Dict[str,np.array]]
        An updated version of the input `cc_arrays`, which contains the assigned tumor IDs.

    """
    for label in cc_arrays.keys():
        cc_files = list(cc_arrays[label].keys())
        tracked_tumors = np.unique(cc_arrays[label][cc_files[0]]).max()

        for i in range(1, len(cc_files)):
            references = [cc_arrays[label][cc_file] for cc_file in cc_files[:i]]
            current = cc_arrays[label][cc_files[i]]

            ccs = [cc for cc in np.unique(current) if cc != 0]

            cc_map = {}

            for cc in ccs:
                cc_mask = (current == cc).astype(int)

                for reference in references:
                    tumor_id = (cc_mask * reference).max()

                    if tumor_id > 0:
                        cc_map[cc] = tumor_id
                        break

                if tumor_id == 0:
                    tracked_tumors += 1
                    cc_map[cc] = tracked_tumors

            vector_map = np.vectorize(lambda x: cc_map.get(x, x))
            current = vector_map(current)

            cc_arrays[label][cc_files[i]] = current

    return cc_arrays


def track_patient_tumors(
    patient_df: pd.DataFrame,
    tracking_dir: Path | str,
    pipeline_key: str = "preprocessed",
    labels: Sequence[int] = [1],
) -> pd.DataFrame:
    """
    Assign tumor IDs in all of the segmentations masks for a single patient.

    Parameters
    ----------
    patient_df: pd.DataFrame
        A DataFrame containing preprocessed segmentation locations and information required
        for the output file names for a single patient. It must contain the columns:
        'AnonPatientID', 'AnonStudyID', 'SeriesType', and f'{pipeline_key}Seg'. Additionally,
        the previous preprocessing is assumed to have been registered longitudinally or to an
        atlas.

    tracking_dir: Path | str
        The directory that will contain the tumor id mask files.

    pipeline_key: str
        The key used in the CSV when preprocessing was performed. Defaults to 'preprocessed'.

    labels: Sequence[int]
        A sequence of the labels included in the segmentation masks.

    Returns
    -------
    pd.DataFrame
        A Dataframe with added columns f'{pipeline_key}_label{label}_ids' for each provided
        label.
    """
    required_columns = [
        "AnonPatientID",
        "AnonStudyID",
        "SeriesType",
        f"{pipeline_key}Seg",
    ]

    check_required_columns(patient_df, required_columns)

    tracking_dir = Path(tracking_dir).resolve()
    rows = patient_df.to_dict("records")

    cc_arrays = {}

    for label in labels:
        cc_arrays[label] = {}
        for i in range(len(rows)):
            seg_file = rows[i][f"{pipeline_key}Seg"]
            array = (GetArrayFromImage(ReadImage(seg_file)) == label).astype(int)

            patient_id = rows[i]["AnonPatientID"]
            study_id = rows[i]["AnonStudyID"]
            series_type = rows[i]["SeriesType"]

            basename = f"{patient_id}_{study_id}_ids_label{label}.nii.gz"

            output_dir = tracking_dir / patient_id / study_id / series_type

            os.makedirs(output_dir, exist_ok=True)

            tumor_id_file = str(output_dir / basename)

            cc_arrays[label][tumor_id_file] = connected_components(array)

    cc_arrays = assign_tumor_ids(cc_arrays)

    for label in labels:
        for i, tumor_id_file in enumerate(cc_arrays[label].keys()):
            seg_file = rows[i][f"{pipeline_key}Seg"]
            seg = ReadImage(seg_file)

            tumor_id_array = cc_arrays[label][tumor_id_file]
            output_nifti = GetImageFromArray(tumor_id_array)

            output_nifti.CopyInformation(seg)
            WriteImage(output_nifti, tumor_id_file)

            rows[i][f"{pipeline_key}_label{label}_ids"] = tumor_id_file

    out_df = pd.DataFrame(rows, dtype=str)

    return out_df


def track_tumors_csv(
    csv: Path | str,
    tracking_dir: Path | str,
    patients: Sequence[str] | None,
    pipeline_key: str = "preprocessed",
    labels: Sequence[int] = [1],
    cpus: int = 1,
) -> pd.DataFrame:
    """
    Assign tumor IDs in all of the segmentations masks for every patient
    within a dataset.

    Parameters
    ----------
    csv: Path | str
        The path to a CSV containing an entire dataset. It must contain the following columns:
        'AnonPatientID', 'AnonStudyID', 'SeriesType', and f'{pipeline_key}Seg'. Additionally,
        the previous preprocessing is assumed to have been registered longitudinally or to an
        atlas.

    tracking_dir: Path | str
        The directory that will contain the tumor id mask files.

    patients: Sequence[str] | None
        A sequence of patients to select from the 'AnonPatientID' column of the CSV. If 'None'
        is provided, all patients will be preprocessed.

    pipeline_key: str
        The key used in the CSV when preprocessing was performed. Defaults to 'preprocessed'.

    labels: Sequence[int]
        A sequence of the labels included in the segmentation masks.

    cpus: int
        Number of cpus to use for multiprocessing. Defaults to 1 (no multiprocessing).

    Returns
    -------
    pd.DataFrame
        A Dataframe with added columns f'{pipeline_key}_label{label}_ids' for each provided
        label. This function will also overwrite the input CSV with this DataFrame.
    """
    df = pd.read_csv(csv, dtype=str)

    required_columns = [
        "AnonPatientID",
        "AnonStudyID",
        "SeriesType",
        f"{pipeline_key}Seg",
    ]

    check_required_columns(df, required_columns)

    tracking_dir = Path(tracking_dir)
    errorfile = tracking_dir / f"{str(datetime.datetime.now()).replace(' ', '_')}.txt"

    filtered_df = df.dropna(subset=[f"{pipeline_key}Seg"])

    if patients is None:
        patients = list(filtered_df["AnonPatientID"].unique())

    kwargs_list = [
        {
            "patient_df": filtered_df[filtered_df["AnonPatientID"] == patient].copy(),
            "tracking_dir": tracking_dir,
            "pipeline_key": pipeline_key,
            "labels": labels,
        }
        for patient in patients
    ]

    with tqdm(
        total=len(kwargs_list), desc="Assigning Tumor IDs"
    ) as pbar, ProcessPoolExecutor(cpus if cpus >= 1 else 1) as executor:
        futures = {
            executor.submit(track_patient_tumors, **kwargs): kwargs
            for kwargs in kwargs_list
        }

        for future in as_completed(futures.keys()):
            try:
                tracked_df = future.result()

            except Exception as error:
                update_errorfile(
                    func_name="preprocessing.qc.track_patient_tumors",
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
            df = pd.merge(df, tracked_df, how="outer")
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

    return df


__all__ = ["track_patient_tumors", "track_tumors_csv"]
