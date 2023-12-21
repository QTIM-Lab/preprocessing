### reorganize dicom data to match BIDS organizational scheme
import pandas as pd
import glob
import os
import shutil
import multiprocessing

from preprocessing.constants import META_KEYS
from pydicom import dcmread
from pathlib import Path
from tqdm import tqdm


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
        Should follow the <Patient_ID>/<Study_ID> convention
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
                "Anon_PatientID": f"sub-{patient.replace('_', '')}",
                "StudyInstanceUID": getattr(
                    dcm, "StudyInstanceUID", None
                ),  # cover edge cases
                "Anon_StudyID": f"ses-{i+1:02d}",
            }

            print(anon_key)
            dicts.append(anon_key)
    df = pd.DataFrame(dicts).sort_values("Anon_PatientID")
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "anonymization_keys.csv")
    print(f"saving to {out_csv}")
    df.to_csv(out_csv, index=False)
    return df


def copy_dicoms(
    sub_dir: Path | str, new_dicom_dir: Path | str, anon_df: pd.DataFrame | None
) -> pd.DataFrame:
    """
    Copies all of the dicoms present within a directory to
    <new_dicom_dir>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm

    Parameters
    __________
    sub_dir: Path | str
        A directory containing DICOM files. Intended to be a subdirectory of within
        the root directory of a DICOM dataset.
    new_dicom_dir: Path | str
        The new root directory under which the copied DICOMs will be stored.
    anon_df: pd.DataFrame
        A DataFrame containing the key to match an Anonymized PatientID and
        Visit_ID to the StudyInstanceUID of the DICOMs.

    Returns
    _______
    pd.DataFrame:
        A DataFrame containing the location and metadata of the DICOM data at the
        series level.
    """
    sub_dir = Path(sub_dir)
    new_dicom_dir = Path(new_dicom_dir)

    files = list(sub_dir.glob("**/*"))
    rows = []

    for file in files:
        try:
            dcm = dcmread(file, stop_before_pixels=True)
            if not hasattr(dcm, "StudyInstanceUID"):
                continue

        except Exception:
            continue

        if anon_df is not None:
            anon_keys = anon_df[(anon_df["StudyInstanceUID"] == dcm.StudyInstanceUID)]
            anon_patient_id = anon_keys.loc[anon_keys.index[0], "Anon_PatientID"]
            anon_study_id = anon_keys.loc[anon_keys.index[0], "Anon_StudyID"]

        else:
            anon_patient_id = None
            anon_study_id = None

        try:
            row = {
                "Anon_PatientID": anon_patient_id,
                "Anon_StudyID": anon_study_id,
            }
        except Exception as error:
            error = f"{file} encountered: {error}\n"
            print(error)
            errorfile = open(new_dicom_dir / "reorganization_errors.txt", "a")
            errorfile.write(error)

            row = {"Anon_PatientID": None, "Anon_StudyID": None}

        for key in META_KEYS:
            attr = getattr(dcm, key, None)
            row[key] = attr

        save_path = new_dicom_dir / dcm.SeriesInstanceUID
        row["dicoms"] = save_path
        try:
            out_file = save_path / f"{dcm.SOPInstanceUID}.dcm"
        except Exception as error:
            error = f"{file} encountered: {error}\n"
            print(error)
            errorfile = open(new_dicom_dir / "reorganization_errors.txt", "a")
            errorfile.write(error)
            # file is likely corrupted, so skip
            continue

        os.makedirs(save_path, exist_ok=True)
        shutil.copy(file, out_file)

        rows.append(row)
    df = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["SeriesInstanceUID"])
        .reset_index(drop=True)
    )
    print(df)
    return df


def copy_dicoms_star(args) -> pd.DataFrame:
    """
    Helper function intended for use only in 'reorganize_dicoms'. Provides an imap
    compatible version of 'copy_dicoms'.
    """
    return copy_dicoms(*args)


def reorganize_dicoms(
    original_dicom_dir: Path | str,
    new_dicom_dir: Path | str,
    anon_csv: Path | str | pd.DataFrame | None,
    cpus: int = 0,
) -> pd.DataFrame:
    """
    Copies all of the dicoms present within a dataset's root directory to
    <new_dicom_dir>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm

    Parameters
    __________
    original_dicom_dir: Path | str
        The original root directory containing all of the DICOM files for a dataset.
    new_dicom_dir: Path | str
        The new root directory under which the copied DICOMs will be stored.
    anon_csv: Path | str | pd.DataFrame | None
        A CSV or DataFrame containing the key to match an Anonymized PatientID and
        Visit_ID to the StudyInstanceUID of the DICOMs. If None is provided, the
        anonymization will be completed automatically based on the DICOM headers.
    cpus: int
        Number of cpus to use for multiprocessing. Defaults to 0 (no multiprocessing).

    Returns
    _______
    pd.DataFrame:
        A DataFrame containing the location and metadata of the DICOM data at the
        series level.
    """
    original_dicom_dir = Path(original_dicom_dir)
    new_dicom_dir = Path(new_dicom_dir)

    if isinstance(anon_csv, (Path, str)):
        anon_df = pd.read_csv(anon_csv)

    elif isinstance(anon_csv, pd.DataFrame):
        anon_df = anon_csv

    else:
        anon_df = None

    sub_dirs = list(original_dicom_dir.glob("*/"))

    if cpus == 0:
        outputs = [
            copy_dicoms(sub_dir, new_dicom_dir, anon_df)
            for sub_dir in tqdm(sub_dirs, desc="Copying DICOMs")
        ]

    else:
        inputs = [[sub_dir, new_dicom_dir, anon_df] for sub_dir in sub_dirs]

        with multiprocessing.Pool(cpus) as pool:
            outputs = list(
                tqdm(
                    pool.imap(copy_dicoms_star, inputs),
                    total=len(sub_dirs),
                    desc="Copying DICOMs",
                )
            )

    if anon_df is not None:
        df = (
            pd.concat(outputs, ignore_index=True)
            .drop_duplicates(subset=["SeriesInstanceUID"])
            .sort_values(["Anon_PatientID", "Anon_StudyID"])
            .reset_index(drop=True)
        )

    else:
        df = (
            pd.concat(outputs, ignore_index=True)
            .drop_duplicates(subset=["SeriesInstanceUID"])
            .sort_values(["StudyDate"])
            .reset_index(drop=True)
        )

        anon_patient_dict = {}
        patients = df["PatientID"].dropna().unique()
        for i, patient in enumerate(patients):
            anon_patient_dict[patient] = f"sub-{i+1:02d}"

        def anonymize_patient(x):
            if isinstance(x, str):
                x = anon_patient_dict[x]
            return x

        df["Anon_PatientID"] = df["PatientID"].apply(anonymize_patient)

        for patient in patients:
            patient_df = df[df["PatientID"] == patient].copy()

            studies = sorted(patient_df["StudyDate"].unique())
            for i, study in enumerate(studies):
                study_df = patient_df[patient_df["StudyDate"] == study].copy()
                study_df["Anon_StudyID"] = [f"ses-{i+1:02d}"] * study_df.shape[0]
                df.update(study_df)

        df = df.sort_values(["Anon_PatientID", "Anon_StudyID"])

    df.to_csv((new_dicom_dir / "dataset.csv"), index=False)

    return df
