import pandas as pd
import datetime

from itertools import islice
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import as_completed, ProcessPoolExecutor
from typing import Literal, Callable, Sequence, Tuple, Dict
from preprocessing.utils import (
    parse_string,
    update_errorfile,
    check_required_columns
)
from preprocessing.constants import META_KEYS
from pydicom.uid import generate_uid


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


def generate_batch(generator, size):
    generator = iter(generator)
    while True:
        batch = list(islice(generator, size))

        if not batch:
            break

        yield batch


def create_dicom_dataset(
    dicom_dir: Path | str,
    reorg_dir: Path | str,
    dataset_csv: Path | str | None = None,
    anon: Literal["auto", "deferred"] = "auto",
    batch_size: int = 20,
    cpus: int = 1
):
    pass



def nifti_batch_processor(
    batch: Sequence[Path],
    file_pattern: str,
    normalized_descriptions: bool = True,
    seg_series: str | None = None,
    seg_target: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    accepted_rows = []
    rejected_rows = []

    for file in batch:
        base_name = file.name.split(".")[0]

        try:
            name_map = parse_string(base_name, file_pattern)
            patient, study, series = [
                name_map[k] for k in ["patient", "study", "series"]
            ]

        except Exception as error:
            rejected_rows.append({
                "Nifti": str(file),
                "RejectionError": str(error)
            })

            continue

        if series == seg_series:
            continue

        row = {
            "PatientID": patient,
            "StudyDate": study,
            "SeriesInstanceUID": generate_uid(),
            "SeriesDescription": series,
        }

        if normalized_descriptions:
            row["NormalizedSeriesDescription"] = series

        row["Nifti"] = str(file)

        if series == seg_target and seg_series is not None:
            seg = file.parent / file.name.replace(series, seg_series)
            if seg.exists():
                row["Seg"] = str(seg)

        accepted_rows.append(row)


    return (pd.DataFrame(accepted_rows), pd.DataFrame(rejected_rows))



def create_nifti_dataset(
    nifti_dir: Path | str,
    dataset_csv: Path | str,
    anon: Literal["is_anon", "auto", "deferred"] = "auto",
    batch_size: int = 20,
    batch_processor: Callable = nifti_batch_processor,
    processor_kwargs: Dict = {
        "file_pattern": "{patient}_{study}_{series}",
        "seg_series": "seg",
        "seg_target": "T1Post"
    },
    cpus: int = 1
):


    nifti_dir = Path(nifti_dir)
    dataset_csv = Path(dataset_csv)
    dataset_csv.parent.mkdir(parents=True, exist_ok=True)
    rejection_csv = str(dataset_csv).replace(".csv", "_rejections.csv")
    errorfile = dataset_csv.parent /  f"{str(datetime.datetime.now()).replace(' ', '_')}.txt"

    path_generator = nifti_dir.glob("**/*.nii*")
    accepted_dfs = []
    rejected_dfs = []

    with tqdm(
        total=0, desc="Constructing NIfTI dataset", dynamic_ncols=True
    ) as pbar, ProcessPoolExecutor(cpus if cpus >= 1 else 1) as executor:
        futures = set()
        future_map = {}

        for batch in generate_batch(path_generator, batch_size):
            kwargs = {"batch": batch, **processor_kwargs}
            future = executor.submit(batch_processor, **kwargs)
            futures.add(future)
            future_map[future] = kwargs


            pbar.total += 1
            pbar.refresh()

            completed_futures = set()

            for future in futures:
                if future.done():
                    try:
                        accepted_df, rejected_df = future.result()

                    except Exception as error:
                        update_errorfile(
                            func_name=f"{batch_processor.__module__}.{batch_processor.__name__}",
                            kwargs=future_map[future],
                            errorfile=errorfile,
                            error=error
                        )

                        completed_futures.add(future)
                        pbar.update(1)
                        continue

                    accepted_dfs.append(accepted_df)
                    rejected_dfs.append(rejected_df)

                    completed_futures.add(future)
                    pbar.update(1)

            futures.difference_update(completed_futures)


        for future in as_completed(futures):
            try:
                accepted_df, rejected_df = future.result()

            except Exception as error:
                update_errorfile(
                    func_name=f"{batch_processor.__module__}.{batch_processor.__name__}",
                    kwargs=future_map[future],
                    errorfile=errorfile,
                    error=error
                )


                pbar.update(1)
                continue

            accepted_dfs.append(accepted_df)
            rejected_dfs.append(rejected_df)
            pbar.update(1)

    df = pd.concat(accepted_dfs)
    discarded_df = pd.concat(rejected_dfs)

    if df.empty:
        print("No matching files have been found and a dataset could not be created")

        return

    study_uid_map = {}

    for patient_id in df["PatientID"].unique():
        patient_df = df[df["PatientID"] == patient_id].copy()

        for study in patient_df["StudyDate"].unique():
            study_uid_map[(patient_id, study)] = generate_uid()

    df["StudyInstanceUID"] = df.apply(
        lambda x: study_uid_map[(x["PatientID"], x["StudyDate"])],
        axis=1
    )

    if anon == "is_anon":
        df["AnonPatientID"] = df["PatientID"]
        df["AnonStudyID"] = df["StudyDate"]
        print("Anonymization completed")

    elif anon == "auto":
        df = anonymize_df(df)
        print("Anonymization completed")

    else:
        print(
            "Anonymization has been skipped. Add the 'AnonPatientID' and "
            "'AnonPatientID' manually before running subsequent commands."
        )

    columns = [key for key in META_KEYS if key in df.keys()]
    columns += [key for key in df.keys() if key not in columns]

    df = df[columns]

    df.to_csv(dataset_csv, index=False)
    print(f"Dataset written to {dataset_csv}")

    if not discarded_df.empty:
        discarded_df.to_csv(rejection_csv, index=False)
        print(
            "Additional NIfTI files were found, but did not follow expected format. "
            f"Rejected files are recorded in {rejection_csv}"
        )

    if Path(errorfile).exists():
        print(f"Errors encountered while running this script and are recorded in {errorfile}")