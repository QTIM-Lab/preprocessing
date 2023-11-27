### reorganize dicom data to match BIDS organizational scheme
import pandas as pd
import glob
import os

from preprocessing.constants import META_KEYS
from pydicom import dcmread
from pathlib import Path


def find_anon_keys(input_dir: Path | str, output_dir: Path | str) -> pd.DataFrame:
    """
    Create anonymization keys for anonymous PatientID and StudyID
    from previous QTIM organizational scheme. Is compatible
    with data following a following <Patient_ID>/<Study_ID> directory
    hierarchy.

    Parameters
    ----------
    input_dir : Path | str
        The directory containing all of the dicom files for a project.
        Should follow the <Patient_ID>/<Study_ID> convention
    output_dir : Path | str
        The directory that will contain the output csv and potentially
        an error file.

    Returns
    -------
    df
        A DataFrame containing the key to match Anonymized PatientID and
        Visit_ID to their DICOM header values. Also saved as a csv within
        the output_dir.

    """
    os.chdir(input_dir)
    dicts = []

    patients = list(glob.glob("*/"))

    for patient in patients:
        patient = patient.replace("/", "")
        os.chdir(os.path.join(input_dir, patient))
        studies = list(glob.glob("*/"))
        for i, study in enumerate(studies):
            study = study.replace("/", "")
            os.chdir(os.path.join(input_dir, patient, study))

            files = list(Path(os.getcwd()).glob("**/*"))
            for file in files:
                dcm_found = False
                try:
                    dcm = dcmread(file, stop_before_pixels=True)
                    if (not hasattr(dcm, "PatientID")) or (
                        not hasattr(dcm, "StudyInstanceUID")
                    ):
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
                "PatientID": getattr(dcm, "PatientID"),
                "Anon_PatientID": patient,
                "StudyInstanceUID": getattr(dcm, "StudyInstanceUID"),
                "Anon_StudyID": f"Visit_{i+1:02d}",
            }

            print(anon_key)
            dicts.append(anon_key)
    df = pd.DataFrame(dicts).sort_values("Anon_PatientID")
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "anonymization_keys.csv")
    print(f"saving to {out_csv}")
    df.to_csv(out_csv, index=False)
    return df


def reorganize_dicoms(
    original_dicom_dir: Path | str,
    new_dicom_dir: Path | str,
    anon_csv: Path | str | pd.DataFrame,
    cpus: int = 0,
) -> None:
    if isinstance(original_dicom_dir, str):
        original_dicom_dir = Path(original_dicom_dir)

    if isinstance(new_dicom_dir, str):
        new_dicom_dir = Path(new_dicom_dir)

    if isinstance(anon_csv, (Path, str)):
        anon_df = pd.read_csv(anon_csv)

    elif isinstance(anon_csv, pd.DataFrame):
        anon_df = anon_csv

    files = list(original_dicom_dir.glob("**/*"))

    for file in files:
        try:
            dcm = dcmread(file)
            if (not hasattr(dcm, "PatientID")) or (
                not hasattr(dcm, "StudyInstanceUID")
            ):
                continue

        except Exception:
            continue

        anon_keys = anon_df[
            (anon_df["PatientID"] == dcm.PatientID)
            & (anon_df["StudyInstanceUID"] == dcm.StudyInstanceUID)
        ]

        row = {
            "Anon_PatientID": anon_keys.loc[anon_keys.index[0], "Anon_PatientID"],
            "Anon_StudyID": anon_keys.loc[anon_keys.index[0], "Anon_StudyID"],
        }

        for key in META_KEYS:
            attr = getattr(dcm, key, None)
            row[key] = attr

        print(row)
        quit()

    # return df
