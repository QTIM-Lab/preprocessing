### reorganize dicom data to match BIDS organizational scheme
import pandas as pd
import glob
import os
import shutil

from preprocessing.constants import META_KEYS
from preprocessing.utils import check_required_columns
from pydicom import dcmread
from pydicom.uid import generate_uid
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


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


def anonymize_df(df: pd.DataFrame, check_columns: bool = True):
    """
    Apply automated anonymization to a DatFrame. This function assumes
    that the 'PatientID' and 'StudyID' tags are consistent and correct
    to derive Anon_PatientID = 'sub_{i:02d}' and Anon_StudyID = 'ses_{i:02d}'.

    Parameters
    __________
    df: pd.DataFrame
        A DataFrame for which you wish to provide anonymized patient and study
        identifiers. It must contain the columns: 'PatientID' and 'StudyDate'.
    check_columns: bool
        Whether to check `df` for required columns. Defaults to True.

    Returns
    _______
    pd.DataFrame
        `df` with added or corrected columns 'Anon_PatientID' and
        'Anon_StudyID'.
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

    df["Anon_PatientID"] = df["PatientID"].apply(
        lambda x: anon_patient_dict[x] if not pd.isna(x) else x
    )

    df["Anon_StudyID"] = df.apply(
        (
            lambda x: anon_study_dict[(x["PatientID"], x["StudyDate"])]
            if not (pd.isna(x["PatientID"]) or pd.isna(x["StudyDate"]))
            else None
        ),
        axis=1,
    )

    df = (
        df.drop_duplicates(subset="SeriesInstanceUID")
        .sort_values(["Anon_PatientID", "Anon_StudyID"])
        .reset_index(drop=True)
    )
    return df


def reorganize_dicoms(
    original_dicom_dir: Path | str,
    new_dicom_dir: Path | str,
    anon_csv: Path | str | pd.DataFrame | None,
    cpus: int = 1,
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
        Number of cpus to use for multiprocessing. Defaults to 1 (no multiprocessing).

    Returns
    _______
    pd.DataFrame:
        A DataFrame containing the location and metadata of the DICOM data at the
        series level.
    """
    original_dicom_dir = Path(original_dicom_dir)
    new_dicom_dir = Path(new_dicom_dir)
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
        }
        for sub_dir in sub_dirs
    ]

    with tqdm(
        total=len(kwargs_list), desc="Copying DICOMs"
    ) as pbar, ThreadPoolExecutor(cpus if cpus >= 1 else 1) as executor:
        futures = [executor.submit(copy_dicoms, **kwargs) for kwargs in kwargs_list]
        for future in as_completed(futures):
            dicom_df = future.result()

            if dataset_csv.exists():
                df = pd.read_csv(dataset_csv, dtype=str)
                df = pd.merge(df, dicom_df, "outer")

            else:
                df = dicom_df

            if anon_df is not None:
                df = (
                    df.drop_duplicates(subset=["SeriesInstanceUID"])
                    .sort_values(["Anon_PatientID", "Anon_StudyID"])
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
        .sort_values(["Anon_PatientID", "Anon_StudyID"])
        .reset_index(drop=True)
    )
    print("Anonymizing CSV:")

    df = anonymize_df(df)
    df.to_csv(dataset_csv, index=False)

    return df


def nifti_anon_csv(
    nifti_dir: Path | str, output_dir: Path | str, normalized_descriptions: bool = False
):
    nifti_dir = Path(nifti_dir)
    output_dir = Path(output_dir)

    rows = []

    study_uid_dict = {}

    for nifti in nifti_dir.glob("**/*.nii.gz"):
        if nifti.name.startswith("."):
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
            "original_nifti": nifti,
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
    if isinstance(anon_csv, (Path, str)):
        anon_df = pd.read_csv(anon_csv, dtype=str)

    elif isinstance(anon_csv, pd.DataFrame):
        anon_df = anon_csv

    required_columns = [
        "Anon_PatientID",
        "Anon_StudyID",
        "PatientID",
        "StudyDate",
        "SeriesInstanceUID",
        "StudyInstanceUID",
        "SeriesDescription",
        "original_nifti",
        "NormalizedSeriesDescription",
    ]

    optional_columns = ["SeriesType"]

    check_required_columns(anon_df, required_columns, optional_columns)

    nifti_dir = Path(nifti_dir)
    dataset_csv = nifti_dir / "dataset.csv"

    def copy_nifti(anon_row: dict) -> pd.DataFrame:
        out_row = {
            "Anon_PatientID": anon_row["Anon_PatientID"],
            "Anon_StudyID": anon_row["Anon_StudyID"],
        }

        for key in META_KEYS + ["NormalizedSeriesDescription"]:
            out_row[key] = anon_row.get(key, None)

        out_row["SeriesType"] = anon_row.get("SeriesType", "anat")

        anon_patient_id = out_row["Anon_PatientID"]
        anon_study_id = out_row["Anon_StudyID"]
        series_type = out_row["SeriesType"]
        normalized_description = out_row["NormalizedSeriesDescription"]

        output_dir = nifti_dir / anon_patient_id / anon_study_id / series_type
        output_dir.mkdir(parents=True, exist_ok=True)

        nifti_basename = (
            f"{anon_patient_id}_{anon_study_id}_{normalized_description}.nii.gz"
        )

        out_row["nifti"] = output_dir / nifti_basename

        shutil.copy(anon_row["original_nifti"], out_row["nifti"])

        return pd.DataFrame([out_row])

    kwargs_list = [
        {
            "anon_row": row,
        }
        for row in anon_df.to_dict("records")
    ]

    with tqdm(
        total=len(kwargs_list), desc="Copying NIfTIs"
    ) as pbar, ThreadPoolExecutor(cpus if cpus >= 1 else 1) as executor:
        futures = [executor.submit(copy_nifti, **kwargs) for kwargs in kwargs_list]
        for future in as_completed(futures):
            nifti_df = future.result()

            if dataset_csv.exists():
                df = pd.read_csv(dataset_csv, dtype=str)
                df = pd.merge(df, nifti_df, "outer")

            else:
                df = nifti_df

            df = (
                df.drop_duplicates(subset=["SeriesInstanceUID"])
                .sort_values(["Anon_PatientID", "Anon_StudyID"])
                .reset_index(drop=True)
            )

            df.to_csv(dataset_csv, index=False)

            pbar.update(1)

    df = (
        pd.read_csv(dataset_csv, dtype=str)
        .drop_duplicates(subset=["SeriesInstanceUID"])
        .sort_values(["Anon_PatientID", "Anon_StudyID"])
        .reset_index(drop=True)
    )

    df.to_csv(dataset_csv, index=False)

    return df
