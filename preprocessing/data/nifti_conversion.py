"""
The `nifti_conversion` module contains tools for converting from DICOM to NIfTI format.

Public Functions
----------------
convert_to_nifti
    Convert a DICOM series to a NIfTI file.

convert_study
    Convert a DICOM study to NIfTI files representing each series.

convert_batch_to_nifti
    Convert a DICOM dataset to NIfTI files representing each series.
"""
import os
import pandas as pd
import numpy as np
import datetime

from subprocess import run
from typing import Literal
from tqdm import tqdm
from pathlib import Path
from preprocessing.utils import (
    source_external_software,
    check_required_columns,
    update_errorfile
)
from preprocessing.dcm_tools import sort_slices, calc_slice_distance
from dicom2nifti import convert_directory
from pydicom import dcmread
from numpy.linalg import norm
from concurrent.futures import ProcessPoolExecutor, as_completed


def dicom_integrity_checks(series_dir: Path | str, eps: float = 1e-3) -> bool:
    """
    Check the integrity of a DICOM series. This includes verification of consistent
    values for the 'ImageOrientationPatient', 'SliceThickness', and 'PixelSpacing'
    tags. 'ImageOrientationPatient' must also form an orthonormal basis.

    Parameters
    ----------
    series_dir: Path | str
        The path to a directory containing a DICOM series. All of the DICOM instances
        are assumed to have the .dcm file extension.

    eps: float
        The absolute error tolerance allowed to pass the integrity checks. Defaults to 1e-3.

    Returns
    -------
    bool
        The success or failure of the integrity checks are represented as True or False
        respectively.
    """
    series_dir = Path(series_dir).resolve()

    files = list(series_dir.glob("**/*.dcm"))

    dcms = []

    for file in files:
        dcms.append(dcmread(file, stop_before_pixels=True))

    dcms = sort_slices(dcms)

    # required metadata is present and consistent
    required_metadata = ["ImageOrientationPatient", "SliceThickness", "PixelSpacing"]

    try:
        for meta in required_metadata:
            meta_0 = getattr(dcms[0], meta)
            for dcm in dcms[1:]:
                if not np.allclose(meta_0, getattr(dcm, meta), atol=eps):
                    return False
    except Exception:
        return False

    # orientaation is an orthonormal basis
    orientation = np.array(dcms[0].ImageOrientationPatient)

    if not np.allclose(norm(orientation[:3]), 1, atol=eps) or not np.allclose(
        norm(orientation[3:]), 1, atol=eps
    ):
        return False

    if not np.isclose(orientation[:3].dot(orientation[3:]), 0, atol=eps):
        return False

    # consistent distance
    dists = []
    for dcm in dcms:
        dists.append(
            calc_slice_distance(dcm.ImageOrientationPatient, dcm.ImagePositionPatient)
        )

    spacing = np.diff(dists)

    if not np.allclose(spacing, spacing.mean(), atol=eps):
        return False

    return True


def convert_to_nifti(
    dicom_dir: Path | str,
    nifti_dir: Path | str,
    anon_patient_id: str,
    anon_study_id: str,
    manufacturer: str,
    normalized_series_description: str,
    subdir: Literal["anat", "func", "dwi"],
    overwrite: bool = False,
    source_software: bool = True,
) -> str | None:
    """
    Convert a DICOM series to a NIfTI file.

    Parameters
    ----------
    dicom_dir: Path | str
        The path to a directory containing all of the DICOM instances in a single series.

    nifti_dir: Path | str
        The root directory under which the converted NIfTI files will be written. Subdirectories
        will be created to follow a BIDS inspired convention.

    anon_patient_id: str
        The anonymized PatientID for the series being converted (e.g., 'sub-01').

    anon_study_id: str
        The anonymized StudyID for the series being converted (e.g., 'ses-01').

    manufacturer: str
        The manufacturer information originally stored in the DICOM header.

    normalized_series_description: str
        The series_description normalized to a consistent value within a dataset.

    subdir: Literal['anat', 'func', 'dwi']
        The subdirectory under the study directory. This represents the modality information for
        BIDS. Currently, the supported options are 'anat', 'func', and 'dwi'.

    overwrite: bool
        Whether to overwrite the NIfTI file if there is already one with the same output name.
        Defaults to False.

    source_software: bool
        Whether to call `source_external_software` to add software required for conversion. Defaults
        to True.

    Returns
    -------
    str | None
        The output name of the NIfTI file if it is successfully created, else None.
    """
    if source_software:
        source_external_software()

    nifti_dir = Path(nifti_dir).resolve()

    output_dir = nifti_dir / anon_patient_id / anon_study_id / subdir
    output_nifti = (
        output_dir
        / f"{anon_patient_id}_{anon_study_id}_{normalized_series_description.replace(' ', '_')}"
    )

    if not isinstance(manufacturer, str):
        manufacturer = "n.a."

    if (
        (not overwrite)
        and (output_nifti.with_suffix(".nii.gz").exists())
        and ("hitachi" not in manufacturer.lower())
    ):
        return str(output_nifti) + ".nii.gz"

    if not dicom_integrity_checks(dicom_dir):
        print(
            f"{dicom_dir} does not pass integrity checks and will not be converted to NIfTI"
        )
        return None

    os.makedirs(output_dir, exist_ok=True)

    command = f"dcm2niix -z y -f {output_nifti.name} -o {output_dir} -b y -w {int(overwrite)} {dicom_dir}"
    print(command)

    run(command, shell=True)

    if "hitachi" in manufacturer.lower():
        # reconvert the cases from hitachi to avoid GEIR, but use dcm2niix originally to generate json
        for file in output_dir.glob("*.nii.gz"):
            os.remove(file)
        convert_directory(dicom_dir, output_dir)

        nifti = list(output_dir.glob("*.nii.gz"))[0]
        os.rename(nifti, output_nifti.with_suffix(".nii.gz"))

    if output_nifti.with_suffix(".nii.gz").exists():
        return str(output_nifti) + ".nii.gz"

    else:
        return None


def convert_study(
    study_df: pd.DataFrame,
    nifti_dir: Path | str,
    overwrite_nifti: bool = False,
    source_software: bool = True,
    check_columns: bool = True,
) -> pd.DataFrame:
    """
    Convert a DICOM study to NIfTI files representing each series.

    Parameters
    ----------
    study_df: pd.DataFrame
        A DataFrame containing data for a single study. It must contain the following
        columns: ['Dicoms', 'AnonPatientID', 'AnonStudyID', 'StudyInstanceUID',
        'Manufacturer', 'NormalizedSeriesDescription', 'SeriesType'].

    nifti_dir: Path | str
        The root directory under which the converted NIfTI files will be written. Subdirectories
        will be created to follow a BIDS inspired convention.

    overwrite: bool
        Whether to overwrite the NIfTI file if there is already one with the same output name.
        Defaults to False.

    source_software: bool
        Whether to call `source_external_software` to add software required for conversion. Defaults
        to True.

    check_columns: bool
        Whether to check 'study_df' for the required columns. Defaults to True.

    Returns
    -------
    pd.DataFrame
        A DataFrame that contains the new column: 'Nifti'
    """
    if source_software:
        source_external_software()

    if check_columns:
        required_columns = [
            "Dicoms",
            "AnonPatientID",
            "AnonStudyID",
            "StudyInstanceUID",
            "Manufacturer",
            "NormalizedSeriesDescription",
            "SeriesType",
        ]

        check_required_columns(study_df, required_columns)

    series_descriptions = study_df["NormalizedSeriesDescription"].unique()

    series_dfs = []
    for series_description in series_descriptions:
        series_df = study_df[
            study_df["NormalizedSeriesDescription"] == series_description
        ].copy()
        output_niftis = []

        for i in range(series_df.shape[0]):
            output_nifti = convert_to_nifti(
                dicom_dir=series_df.loc[series_df.index[i], "Dicoms"],
                nifti_dir=nifti_dir,
                anon_patient_id=series_df.loc[series_df.index[i], "AnonPatientID"],
                anon_study_id=series_df.loc[series_df.index[i], "AnonStudyID"],
                manufacturer=series_df.loc[series_df.index[i], "Manufacturer"],
                normalized_series_description=series_description,
                subdir=series_df.loc[series_df.index[i], "SeriesType"],
                overwrite=overwrite_nifti,
                source_software=False,
            )
            output_niftis.append(output_nifti)

        series_df["Nifti"] = output_niftis
        series_dfs.append(series_df)

    df = pd.concat(series_dfs)

    return df


def convert_batch_to_nifti(
    nifti_dir: Path | str,
    csv: Path | str,
    overwrite_nifti: bool = False,
    cpus: int = 1,
    check_columns: bool = True,
) -> pd.DataFrame:
    """
    Convert a DICOM dataset to NIfTI files representing each series.

    Parameters
    ----------
    nifti_dir: Path | str
        The root directory under which the converted NIfTI files will be written. Subdirectories
        will be created to follow a BIDS inspired convention.

    csv: Path | str
        The path to a CSV containing an entire dataset. It must contain the following
        columns: ['Dicoms', 'AnonPatientID', 'AnonStudyID', 'StudyInstanceUID',
        'SeriesInstanceUID', 'Manufacturer', 'NormalizedSeriesDescription', 'SeriesType'].

    overwrite: bool
        Whether to overwrite the NIfTI file if there is already one with the same output name.
        Defaults to False.

    source_software: bool
        Whether to call `source_external_software` to add software required for conversion. Defaults
        to True.

    cpus: int
        Number of cpus to use for multiprocessing. Defaults to 1 (no multiprocessing).

    check_columns: bool
        Whether to check the CSV for the required columns. Defaults to True.

    Returns
    -------
    pd.DataFrame
        A DataFrame that contains the new column: 'Nifti'. This DataFrame will be used to overwrite
        the CSV.
    """

    source_external_software()

    df = pd.read_csv(csv, dtype=str)

    if check_columns:
        required_columns = [
            "Dicoms",
            "AnonPatientID",
            "AnonStudyID",
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "Manufacturer",
            "NormalizedSeriesDescription",
            "SeriesType",
        ]

        check_required_columns(df, required_columns)

    filtered_df = df.copy().dropna(subset="NormalizedSeriesDescription")

    study_uids = filtered_df["StudyInstanceUID"].unique()

    nifti_dir = Path(nifti_dir).resolve()
    errorfile = nifti_dir / f"{str(datetime.datetime.now()).replace(' ', '_')}.txt"

    kwargs_list = [
        {
            "study_df": filtered_df[
                filtered_df["StudyInstanceUID"] == study_uid
            ].copy(),
            "nifti_dir": nifti_dir,
            "overwrite_nifti": overwrite_nifti,
            "source_software": False,
            "check_columns": False,
        }
        for study_uid in study_uids
    ]

    with tqdm(
        total=len(kwargs_list), desc="Converting to NIfTI"
    ) as pbar, ProcessPoolExecutor(cpus if cpus >= 1 else 1) as executor:
        futures = {
            executor.submit(convert_study, **kwargs): kwargs
            for kwargs in kwargs_list
        }

        for future in as_completed(futures.keys()):
            try:
                nifti_df = future.result()

            except Exception as error:
                update_errorfile(
                    func_name="preprocessing.data.convert_study",
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
            df = pd.merge(df, nifti_df, how="outer")
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


__all__ = ["convert_to_nifti", "convert_study", "convert_batch_to_nifti"]
