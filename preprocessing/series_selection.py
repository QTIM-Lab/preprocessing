import pandas as pd
import multiprocessing

from pathlib import Path
from pydicom import dcmread
from tqdm import tqdm
from mr_series_selection.series_selection import get_series_classification
from preprocessing.utils import check_required_columns

default_key = {
    "T1Pre": [["iso3D AX T1 NonContrast", "iso3D AX T1 NonContrast RFMT"], "anat"],
    "T1Post": [["iso3D AX T1 WithContrast", "iso3D AX T1 WithContrast RFMT"], "anat"],
}


def description_sort(column):
    string_priority = {"mprage": 0, "bravo": 0, "high_res": 1, "high-res": 1}

    priority_values = []

    for description in column:
        found = False
        for string, priority in string_priority.items():
            if string in description.lower():
                priority_values.append(priority)
                found = True
                break
        if not found:
            priority_values.append(len(string_priority) + 1)

    return pd.Series(priority_values)


def series_in_study(
    study_df: pd.DataFrame,
    ruleset: str = "brain",
    description_key: dict = default_key,
    check_columns: bool = True,
) -> pd.DataFrame:
    if check_columns:
        required_columns = ["StudyInstanceUID", "SeriesDescription"]

        check_required_columns(study_df, required_columns)

    filtered_df = (
        study_df.copy()
        .dropna(subset="SeriesDescription")
        .sort_values(["SeriesDescription"], key=description_sort)
    )

    # skip manual predictions
    skip_keys = []
    if "NormalizedSeriesDescription" in filtered_df.keys():
        for description in filtered_df["NormalizedSeriesDescription"].dropna().unique():
            if description in description_key:
                skip_keys.append(description)
                filtered_df = filtered_df[
                    ~filtered_df["NormalizedSeriesDescription"].str.contains(
                        description, na=False
                    )
                ]

    normalized_descriptions = []
    series_types = []
    for dicom_dir in filtered_df["dicoms"]:
        files = Path(dicom_dir).glob("*")
        dcms = []
        for file in files:
            try:
                dcms.append(dcmread(file))
            except Exception:
                continue

        classification = get_series_classification(ruleset, dcms)
        normalized_description = classification.get("NormalizedDescription", None)

        found = False
        for key in description_key.keys():
            if normalized_description in description_key[key][0]:
                if (key not in normalized_descriptions) and (key not in skip_keys):
                    normalized_descriptions.append(key)
                    series_types.append(description_key[key][1])
                else:
                    # Only predict one of each key for a study
                    normalized_descriptions.append(None)
                    series_types.append(None)
                found = True
                break

        if not found:
            normalized_descriptions.append(None)
            series_types.append(None)

    filtered_df["NormalizedSeriesDescription"] = normalized_descriptions
    filtered_df["SeriesType"] = series_types

    # print(normalized_descriptions)
    # print(list(filtered_df["SeriesDescription"]))
    return filtered_df


def series_in_study_star(args):
    return series_in_study(*args)


def series_from_csv(
    csv: Path | str,
    ruleset: str = "brain",
    description_key: dict = default_key,
    cpus: int = 0,
    check_columns: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(csv, dtype=str)

    if check_columns:
        required_columns = ["StudyInstanceUID", "SeriesDescription"]

        check_required_columns(df, required_columns)

    filtered_df = df.copy().dropna(subset="StudyInstanceUID")
    study_uids = filtered_df["StudyInstanceUID"].unique()

    if cpus == 0:
        outputs = [
            series_in_study(
                filtered_df[filtered_df["StudyInstanceUID"] == study_uid].copy(),
                ruleset,
                description_key,
                False,
            )
            for study_uid in tqdm(study_uids, "Predicting on studies")
        ]

    else:
        inputs = [
            [
                filtered_df[filtered_df["StudyInstanceUID"] == study_uid].copy(),
                ruleset,
                description_key,
                False,
            ]
            for study_uid in study_uids
        ]

        with multiprocessing.Pool(cpus) as pool:
            outputs = list(
                tqdm(
                    pool.imap(series_in_study_star, inputs),
                    total=len(study_uids),
                    desc="Predicting on studies",
                )
            )

    series_df = pd.concat(outputs)
    df = pd.merge(df, series_df, how="outer")
    df = df.sort_values(["Anon_PatientID", "Anon_StudyID"]).reset_index(drop=True)
    df.to_csv(csv, index=False)
    return df


__all__ = [
    "default_key",
    "series_in_study",
    "series_from_csv",
]
