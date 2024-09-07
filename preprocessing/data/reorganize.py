"""
The `reorganize` module provides code for reorganizing DICOM or NIfTI datasets
to follow the BIDS naming conventions. These outputs yield datasets that are
compatible with the rest of the `preprocessing` library.

Public Functions
----------------
find_anon_keys
    Create anonymization keys for anonymous PatientID and StudyID from previous
    QTIM organizational scheme. Is compatible with data following a following
    <Patient_ID>/<Study_ID> directory hierarchy.

nifti_anon_csv
    Create anonymization keys for a dataset that starts within NIfTI format. If the
    'SeriesDescription's are not normalized, 'NormalizedSeriesDescription's must be
    obtained externally before the NIfTI dataset can be reorganized.

reorganize_dicoms
    Reorganize DICOMs to follow a BIDS inspired convention. Any DICOMs found recursively
    within this directory will be reorganized (at least one level of subdirectories
    is assumed). Anonomyzation keys for PatientIDs and StudyIDs are provided within
    a CSV.

reorganize_niftis
    Reorganize a NIfTI dataset to follow a BIDS inspired convention. As NIfTI files lack metadata,
    anonymization keys must be provided in the form of a CSV, such as one obtained with
    `nifti_anon_csv`.
"""
import pandas as pd
import glob
import os
import shutil
import numpy as np
import datetime

from preprocessing.constants import META_KEYS
from preprocessing.utils import check_required_columns, update_errorfile
from pydicom import dcmread
from pydicom.uid import generate_uid
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def find_anon_keys(input_dir: Path | str, output_dir: Path | str) -> pd.DataFrame:
    """
    Create anonymization keys for anonymous PatientID and StudyID
    from previous QTIM organizational scheme. Is compatible
    with data following a following <Patient_ID>/<Study_ID> directory
    hierarchy.

    Parameters
    ----------
    input_dir: Path | str
        The directory containing all of the dicom files for a project.
        Should follow the <Patient_ID>/<Study_ID> convention.

    output_dir: Path | str
        The directory that will contain the output csv and potentially
        an error file.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the key to match an Anonymized PatientID and
        Visit_ID to the StudyInstanceUID of the DICOMs. Also saved as a CSV within
        the 'output_dir'.
    """
    os.chdir(input_dir)
    dicts = []

    patients = sorted(list(glob.glob("*/")))

    for patient in patients:
        patient = patient.replace("/", "")
        os.chdir(os.path.join(input_dir, patient))
        studies = sorted(list(glob.glob("*/")))
        for i, study in enumerate(studies):
            study = study.replace("/", "")
            os.chdir(os.path.join(input_dir, patient, study))

            files = list(Path(os.getcwd()).glob("**/*"))
            for file in files:
                dcm_found = False
                try:
                    dcm = dcmread(file, stop_before_pixels=True)
                    if not hasattr(dcm, "StudyInstanceUID"):
                        continue
                    dcm_found = True
                    break

                except Exception:
                    continue

            if not dcm_found:
                error = (
                    f"No dcms found for ({patient}, {study}) or dcms lack metadata\n"
                )
                print(
                    f"{error}"
                    f"writing to {os.path.join(output_dir, 'anon_key_errors.txt')} "
                )
                os.makedirs(output_dir, exist_ok=True)
                errorfile = open(os.path.join(output_dir, "anon_key_errors.txt"), "a")
                errorfile.write(error)
                continue

            anon_key = {
                "AnonPatientID": f"sub-{patient.replace('_', '')}",
                "StudyInstanceUID": getattr(
                    dcm, "StudyInstanceUID", None
                ),  # cover edge cases
                "AnonStudyID": f"ses-{i+1:02d}",
            }

            print(anon_key)
            dicts.append(anon_key)
    df = pd.DataFrame(dicts).sort_values("AnonPatientID")
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "anonymization_keys.csv")
    print(f"saving to {out_csv}")
    df.to_csv(out_csv, index=False)
    return df


def copy_dicoms(
    sub_dir: Path | str,
    new_dicom_dir: Path | str,
    anon_df: pd.DataFrame | None,
    drop_incomplete_series: bool = True,
) -> pd.DataFrame:
    """
    Copies all of the DICOMs present within a directory to
    <new_dicom_dir>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm.

    Parameters
    ----------
    sub_dir: Path | str
        A directory containing DICOM files. Intended to be a subdirectory of within
        the root directory of a DICOM dataset.

    new_dicom_dir: Path | str
        The new root directory under which the copied DICOMs will be stored.

    anon_df: pd.DataFrame
        A DataFrame containing the key to match an Anonymized PatientID and
        Visit_ID to the StudyInstanceUID of the DICOMs.

    Returns
    -------
    pd.DataFrame:
        A DataFrame containing the location and metadata of the DICOM data at the
        series level.
    """
    sub_dir = Path(sub_dir).resolve()
    new_dicom_dir = Path(new_dicom_dir).resolve()

    files = list(sub_dir.glob("**/*"))
    rows = []
    incomplete_series = []

    for file in files:
        try:
            dcm = dcmread(file, stop_before_pixels=True)
            if not all(
                hasattr(dcm, attr)
                for attr in ["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"]
            ):
                continue

        except Exception:
            continue

        if anon_df is not None:
            anon_keys = anon_df[(anon_df["StudyInstanceUID"] == dcm.StudyInstanceUID)]
            anon_patient_id = anon_keys.loc[anon_keys.index[0], "AnonPatientID"]
            anon_study_id = anon_keys.loc[anon_keys.index[0], "AnonStudyID"]

        else:
            anon_patient_id = None
            anon_study_id = None

        row = {
            "AnonPatientID": anon_patient_id,
            "AnonStudyID": anon_study_id,
        }

        for key in META_KEYS:
            attr = getattr(dcm, key, None)
            row[key] = attr

        save_path = new_dicom_dir / dcm.SeriesInstanceUID
        row["Dicoms"] = save_path

        out_file = save_path / f"{dcm.SOPInstanceUID}.dcm"

        os.makedirs(save_path, exist_ok=True)


        shutil.copy(file, out_file)

        rows.append(row)

    df = pd.DataFrame(rows)

    if drop_incomplete_series:
        for series_uid in incomplete_series:
            save_paths = df[df["SeriesInstanceUID"] == series_uid]["dicoms"]

            for path in save_paths:
                shutil.rmtree(path, ignore_errors=True)
            df = df[~df["SeriesInstanceUID"] == series_uid]

    df = df.drop_duplicates(subset=["SeriesInstanceUID"]).reset_index(drop=True)
    print(df)
    return df


def anonymize_df(df: pd.DataFrame, check_columns: bool = True):
    """
    Apply automated anonymization to a DatFrame. This function assumes
    that the 'PatientID' and 'StudyID' tags are consistent and correct
    to derive AnonPatientID = 'sub_{i:02d}' and AnonStudyID = 'ses_{i:02d}'.

    Parameters
    ----------
    df: pd.DataFrame
        A DataFrame for which you wish to provide anonymized patient and study
        identifiers. It must contain the columns: 'PatientID' and 'StudyDate'.

    check_columns: bool
        Whether to check `df` for required columns. Defaults to True.

    Returns
    -------
    pd.DataFrame
        `df` with added or corrected columns 'AnonPatientID' and
        'AnonStudyID'.
    """

    if check_columns:
        required_columns = [
            "PatientID",
            "StudyDate",
            "SeriesInstanceUID",
        ]

        check_required_columns(df, required_columns)

    anon_patient_dict = {}
    anon_study_dict = {}
    patients = df["PatientID"].dropna().unique()
    for i, patient in enumerate(patients):
        anon_patient_dict[patient] = f"sub-{i+1:02d}"

        patient_df = df[df["PatientID"] == patient].copy()
        study_dates = sorted(patient_df["StudyDate"].unique())
        for j, study_date in enumerate(study_dates):
            anon_study_dict[(patient, study_date)] = f"ses-{j+1:02d}"

    df["AnonPatientID"] = df["PatientID"].apply(
        lambda x: anon_patient_dict[x] if not pd.isna(x) else x
    )

    df["AnonStudyID"] = df.apply(
        (
            lambda x: anon_study_dict[(x["PatientID"], x["StudyDate"])]
            if not (pd.isna(x["PatientID"]) or pd.isna(x["StudyDate"]))
            else None
        ),
        axis=1,
    )

    df = (
        df.drop_duplicates(subset="SeriesInstanceUID")
        .sort_values(["AnonPatientID", "AnonStudyID"])
        .reset_index(drop=True)
    )
    return df


def reorganize_dicoms(
    original_dicom_dir: Path | str,
    new_dicom_dir: Path | str,
    anon_csv: Path | str | pd.DataFrame | None,
    cpus: int = 1,
    drop_incomplete_series: bool = True,
) -> pd.DataFrame:
    """
    Reorganize DICOMs to follow a BIDS inspired convention. Any DICOMs found recursively
    within this directory will be reorganized (at least one level of subdirectories
    is assumed). Anonomyzation keys for PatientIDs and StudyIDs are provided within
    a CSV.

    Parameters
    ----------
    original_dicom_dir: Path | str
        The original root directory containing all of the DICOM files for a dataset.

    new_dicom_dir: Path | str
        The new root directory under which the copied DICOMs will be stored.

    anon_csv: Path | str | pd.DataFrame | None
        A CSV or DataFrame containing the key to match an Anonymized PatientID and
        Visit_ID to the StudyInstanceUID of the DICOMs. If None is provided, the
        anonymization will be completed automatically based on the DICOM headers.

    cpus: int
        Number of cpus to use for multiprocessing. Defaults to 1 (no multiprocessing).

    Returns
    -------
    pd.DataFrame:
        A DataFrame containing the location and metadata of the DICOM data at the
        series level.
    """
    original_dicom_dir = Path(original_dicom_dir).resolve()
    new_dicom_dir = Path(new_dicom_dir).resolve()
    errorfile = new_dicom_dir / f"{str(datetime.datetime.now()).replace(' ', '_')}.txt"
    dataset_csv = new_dicom_dir / "dataset.csv"

    if isinstance(anon_csv, (Path, str)):
        anon_df = pd.read_csv(anon_csv, dtype=str)

    elif isinstance(anon_csv, pd.DataFrame):
        anon_df = anon_csv

    else:
        anon_df = None

    sub_dirs = list(original_dicom_dir.glob("*/"))

    kwargs_list = [
        {
            "sub_dir": sub_dir,
            "new_dicom_dir": new_dicom_dir,
            "anon_df": anon_df,
            "drop_incomplete_series": drop_incomplete_series,
        }
        for sub_dir in sub_dirs
    ]

    with tqdm(
        total=len(kwargs_list), desc="Copying DICOMs"
    ) as pbar, ProcessPoolExecutor(cpus if cpus >= 1 else 1) as executor:
        futures = {
            executor.submit(copy_dicoms, **kwargs): kwargs
            for kwargs in kwargs_list
        }

        for future in as_completed(futures.keys()):
            try:
                dicom_df = future.result()

            except Exception as error:
                update_errorfile(
                    func_name="preprocessing.data.copy_dicoms",
                    kwargs=futures[future],
                    errorfile=errorfile,
                    error=error
                )

                pbar.update(1)
                continue

            if dataset_csv.exists():
                df = pd.read_csv(dataset_csv, dtype=str)
                if dicom_df.empty:
                    pbar.update(1)
                    continue

                df = pd.merge(df, dicom_df, "outer")

            else:
                df = dicom_df

            if anon_df is not None:
                df = (
                    df.drop_duplicates(subset=["SeriesInstanceUID"])
                    .sort_values(["AnonPatientID", "AnonStudyID"])
                    .reset_index(drop=True)
                )

                df.to_csv(dataset_csv, index=False)

            else:
                df = df.drop_duplicates(subset=["SeriesInstanceUID"]).reset_index(
                    drop=True
                )
                df = anonymize_df(df, False)
                df.to_csv(dataset_csv, index=False)

            pbar.update(1)

    df = (
        pd.read_csv(dataset_csv, dtype=str)
        .drop_duplicates(subset=["SeriesInstanceUID"])
        .sort_values(["AnonPatientID", "AnonStudyID"])
        .reset_index(drop=True)
    )
    print("Anonymizing CSV:")

    df = anonymize_df(df)
    df.to_csv(dataset_csv, index=False)

    return df


def nifti_anon_csv(
    nifti_dir: Path | str, output_dir: Path | str, normalized_descriptions: bool = False
) -> pd.DataFrame:
    """
    Create anonymization keys for a dataset that starts within NIfTI format. If the
    'SeriesDescription's are not normalized, 'NormalizedSeriesDescription's must be
    obtained externally before the NIfTI dataset can be reorganized.

    Parameters
    ----------
    nifti_dir: Path | str
        The directory containing all of the NIfTI files you wish to anonymize.

    output_dir: Path | str
        The directory that will contain the output CSV and potentially an error file.

    normalized_descriptions: bool
        Whether the 'SeriesDescription' in the NIfTI file name is already normalized.
        Defaults to False.

    Returns
    -------
        A DataFrame containing the key to match simulated DICOM metadata to the original
        NIfTI files. Also saved as a CSV within the 'output_dir'.
    """
    nifti_dir = Path(nifti_dir).resolve()
    output_dir = Path(output_dir).resolve()

    rows = []

    study_uid_dict = {}

    for nifti in nifti_dir.glob("**/*.nii.gz"):
        if nifti.name.startswith("."):
            continue

        seg_identifiers = ["label", "seg"]
        if np.any([i in nifti.name.lower() for i in seg_identifiers]):
            continue

        name_components = nifti.name.replace(".nii.gz", "").split("-")

        try:
            patient_id = name_components[-3]
            study_date = name_components[-2]
            series_description = name_components[-1]

        except Exception:
            error = (
                f"{nifti} does not follow the expected convention for this command. "
                "Rename the files to follow a `<PatientIdentifier>-<StudyDate | StudyIdentifier>-<SeriesDescription>.nii.gz` "
                "naming convention. Names that include a leading component separated with '-' should also "
                "be compatible.\n"
            )

            print(error)
            with open(output_dir / "nifti_anon_errors.txt", "a") as e:
                e.write(error)
            continue

        if (patient_id, study_date) not in study_uid_dict:
            study_uid = generate_uid()
            study_uid_dict[(patient_id, study_date)] = study_uid

        else:
            study_uid = study_uid_dict[(patient_id, study_date)]

        row = {
            "PatientID": patient_id,
            "StudyDate": study_date,
            "SeriesInstanceUID": generate_uid(),
            "StudyInstanceUID": study_uid,
            "SeriesDescription": series_description,
            "OriginalNifti": nifti,
        }

        if normalized_descriptions:
            row["NormalizedSeriesDescription"] = series_description

        rows.append(row)

    df = pd.DataFrame(rows)
    df = anonymize_df(df)

    df.to_csv(output_dir / "nifti_anon.csv", index=False)
    return df


def reorganize_niftis(
    nifti_dir: Path | str,
    anon_csv: Path | str | pd.DataFrame,
    cpus: int = 1,
) -> pd.DataFrame:
    """
    Reorganize a NIfTI dataset to follow a BIDS inspired convention. As NIfTI files lack metadata,
    anonymization keys must be provided in the form of a CSV, such as one obtained with
    `nifti_anon_csv`.

    Parameters
    ----------
    nifti_dir: Path | str
        The new root directory under which the copied NIfTIs will be stored.

    anon_csv: Path | str | pd.DataFrame
        A CSV containing the original location of NIfTI files and metadata required for
        preprocessing commands. It must contain the columns: 'AnonPatientID',
        'AnonStudyID', 'PatientID', 'StudyDate', 'SeriesInstanceUID', 'StudyInstanceUID',
        'SeriesDescription', 'OriginalNifti', and 'NormalizedSeriesDescription'. 'SeriesType'
        can also be provided, otherwise "anat" will be assumed.

    cpus: int
         Number of cpus to use for multiprocessing. Defaults to 1 (no multiprocessing).

    Returns
    -------
    pd.DataFrame:
        A DataFrame containing the location and simulated metadata of the NIfTI data at the
        series level. Also saved as a CSV within the 'nifti_dir'.
    """
    if isinstance(anon_csv, (Path, str)):
        anon_df = pd.read_csv(anon_csv, dtype=str)

    elif isinstance(anon_csv, pd.DataFrame):
        anon_df = anon_csv

    required_columns = [
        "AnonPatientID",
        "AnonStudyID",
        "PatientID",
        "StudyDate",
        "SeriesInstanceUID",
        "StudyInstanceUID",
        "SeriesDescription",
        "OriginalNifti",
        "NormalizedSeriesDescription",
    ]

    optional_columns = ["SeriesType", "Seg"]

    check_required_columns(anon_df, required_columns, optional_columns)

    nifti_dir = Path(nifti_dir).resolve()
    errorfile = nifti_dir / f"{str(datetime.datetime.now()).replace(' ', '_')}.txt"
    dataset_csv = nifti_dir / "dataset.csv"

    def copy_nifti(anon_row: dict) -> pd.DataFrame:
        out_row = {
            "AnonPatientID": anon_row["AnonPatientID"],
            "AnonStudyID": anon_row["AnonStudyID"],
        }

        for key in META_KEYS + ["NormalizedSeriesDescription"]:
            out_row[key] = anon_row.get(key, None)

        out_row["SeriesType"] = anon_row.get("SeriesType", "anat")

        anon_patient_id = out_row["AnonPatientID"]
        anon_study_id = out_row["AnonStudyID"]
        series_type = out_row["SeriesType"]
        normalized_description = out_row["NormalizedSeriesDescription"]

        output_dir = nifti_dir / anon_patient_id / anon_study_id / series_type
        output_dir.mkdir(parents=True, exist_ok=True)

        nifti_basename = (
            f"{anon_patient_id}_{anon_study_id}_{normalized_description}.nii.gz"
        )

        out_row["Nifti"] = output_dir / nifti_basename

        shutil.copy(anon_row["OriginalNifti"], out_row["Nifti"])

        if "Seg" in anon_row and not pd.isna(anon_row["Seg"]):
            seg_basename = f"{anon_patient_id}_{anon_study_id}_seg.nii.gz"

            out_row["Seg"] = output_dir / seg_basename

            shutil.copy(anon_row["Seg"], out_row["Seg"])

        return pd.DataFrame([out_row])

    kwargs_list = [
        {
            "anon_row": row,
        }
        for row in anon_df.to_dict("records")
    ]

    with tqdm(
        total=len(kwargs_list), desc="Copying NIfTIs"
    ) as pbar, ProcessPoolExecutor(cpus if cpus >= 1 else 1) as executor:
        futures = {
            executor.submit(copy_nifti, **kwargs): kwargs
            for kwargs in kwargs_list
        }

        for future in as_completed(futures.keys()):
            try:
                nifti_df = future.result()

            except Exception as error:
                update_errorfile(
                    func_name="preprocessing.data.copy_nifti",
                    kwargs=futures[future],
                    errorfile=errorfile,
                    error=error
                )

                pbar.update(1)
                continue

            if dataset_csv.exists():
                df = pd.read_csv(dataset_csv, dtype=str)
                df = pd.merge(df, nifti_df, "outer")

            else:
                df = nifti_df

            df = (
                df.drop_duplicates(subset=["SeriesInstanceUID"])
                .sort_values(["AnonPatientID", "AnonStudyID"])
                .reset_index(drop=True)
            )

            df.to_csv(dataset_csv, index=False)

            pbar.update(1)

    df = (
        pd.read_csv(dataset_csv, dtype=str)
        .drop_duplicates(subset=["SeriesInstanceUID"])
        .sort_values(["AnonPatientID", "AnonStudyID"])
        .reset_index(drop=True)
    )

    df.to_csv(dataset_csv, index=False)

    return df


__all__ = [
    "find_anon_keys",
    "nifti_anon_csv",
    "reorganize_dicoms",
    "reorganize_niftis",
]
