"""
The `tumor_ids` module defines the tools for longitudinal tracking of
individual tumors. Each connected component for a given label within a segmentation mask
is assigned unique ID that will remain consistent across all scans belonging
to the same patient patient. This code assumes that the longitudinal or atlas registration
was used when preprocessing the data.

Public Functions
________________
track_patient_tumors
    Assign tumor IDs in all of the segmentations masks for a single patient.

track_tumors_csv
    Assign tumor IDs in all of the segmentations masks for every patient within a dataset.
"""
import pandas as pd
import numpy as np
import os

from preprocessing.utils import check_required_columns
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
    __________
    cc_arrays: Dict[str, Dict[str, np.array]]
        A nested dictionary mapping the current label and corresponding file path to an
        array containing connected components.


    Returns
    _______
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
    __________
    patient_df: pd.DataFrame
        A DataFrame containing preprocessed segmentation locations and information required
        for the output file names for a single patient. It must contain the columns:
        'Anon_PatientID', 'Anon_StudyID', 'SeriesType', and f'{pipeline_key}_seg'. Additionally,
        the previous preprocessing is assumed to have been registered longitudinally or to an
        atlas.

    tracking_dir: Path | str
        The directory that will contain the tumor id mask files.

    pipeline_key: str
        The key used in the CSV when preprocessing was performed. Defaults to 'preprocessed'.

    labels: Sequence[int]
        A sequence of the labels included in the segmentation masks.

    Returns
    _______
    pd.DataFrame
        A Dataframe with added columns f'{pipeline_key}_label{label}_ids' for each provided
        label.
    """
    required_columns = [
        "Anon_PatientID",
        "Anon_StudyID",
        "SeriesType",
        f"{pipeline_key}_seg",
    ]

    check_required_columns(patient_df, required_columns)

    tracking_dir = Path(tracking_dir)
    rows = patient_df.to_dict("records")

    cc_arrays = {}

    for label in labels:
        cc_arrays[label] = {}
        for i in range(len(rows)):
            seg_file = rows[i][f"{pipeline_key}_seg"]
            array = (GetArrayFromImage(ReadImage(seg_file)) == label).astype(int)

            patient_id = rows[i]["Anon_PatientID"]
            study_id = rows[i]["Anon_StudyID"]
            series_type = rows[i]["SeriesType"]

            basename = f"{patient_id}_{study_id}_ids_label{label}.nii.gz"

            output_dir = tracking_dir / patient_id / study_id / series_type

            os.makedirs(output_dir, exist_ok=True)

            tumor_id_file = str(output_dir / basename)

            cc_arrays[label][tumor_id_file] = connected_components(array)

    cc_arrays = assign_tumor_ids(cc_arrays)

    for label in labels:
        for i, tumor_id_file in enumerate(cc_arrays[label].keys()):
            seg_file = rows[i][f"{pipeline_key}_seg"]
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
    __________
    csv: Path | str
        The path to a CSV containing an entire dataset. It must contain the following columns:
        'Anon_PatientID', 'Anon_StudyID', 'SeriesType', and f'{pipeline_key}_seg'. Additionally,
        the previous preprocessing is assumed to have been registered longitudinally or to an
        atlas.

    tracking_dir: Path | str
        The directory that will contain the tumor id mask files.

    patients: Sequence[str] | None
        A sequence of patients to select from the 'Anon_PatientID' column of the CSV. If 'None'
        is provided, all patients will be preprocessed.

    pipeline_key: str
        The key used in the CSV when preprocessing was performed. Defaults to 'preprocessed'.

    labels: Sequence[int]
        A sequence of the labels included in the segmentation masks.

    cpus: int
        Number of cpus to use for multiprocessing. Defaults to 1 (no multiprocessing).

    Returns
    _______
    pd.DataFrame
        A Dataframe with added columns f'{pipeline_key}_label{label}_ids' for each provided
        label. This function will also overwrite the input CSV with this DataFrame.
    """
    df = pd.read_csv(csv, dtype=str)

    required_columns = [
        "Anon_PatientID",
        "Anon_StudyID",
        "SeriesType",
        f"{pipeline_key}_seg",
    ]

    check_required_columns(df, required_columns)

    filtered_df = df.dropna(subset=[f"{pipeline_key}_seg"])

    if patients is None:
        patients = list(filtered_df["Anon_PatientID"].unique())

    kwargs_list = [
        {
            "patient_df": filtered_df[filtered_df["Anon_PatientID"] == patient].copy(),
            "tracking_dir": tracking_dir,
            "pipeline_key": pipeline_key,
            "labels": labels,
        }
        for patient in patients
    ]

    with tqdm(
        total=len(kwargs_list), desc="Assigning Tumor IDs"
    ) as pbar, ProcessPoolExecutor(cpus if cpus >= 1 else 1) as executor:
        futures = [
            executor.submit(track_patient_tumors, **kwargs) for kwargs in kwargs_list
        ]
        for future in as_completed(futures):
            tracked_df = future.result()
            df = (
                pd.read_csv(csv, dtype=str)
                .drop_duplicates(subset="SeriesInstanceUID")
                .reset_index(drop=True)
            )
            df = pd.merge(df, tracked_df, how="outer")
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

    return df


__all__ = ["track_patient_tumors", "track_tumors_csv"]
