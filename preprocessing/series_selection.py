import pandas as pd

from pathlib import Path
from pydicom import dcmread
from tqdm import tqdm
from mr_series_selection.series_selection import get_series_classification
from preprocessing.utils import check_required_columns
from concurrent.futures import ThreadPoolExecutor, as_completed

default_key = {
    "T1Pre": [["iso3D AX T1 NonContrast", "iso3D AX T1 NonContrast RFMT"], "anat"],
    "T1Post": [["iso3D AX T1 WithContrast", "iso3D AX T1 WithContrast RFMT"], "anat"],
}


def description_sort(column: pd.Series) -> pd.Series:
    """
    Sort a pd.Series containing the original SeriesDescription values and prioritize
    relevant substrings. Intended only for use within 'series_in_study' as a custom
    sort function.

    Parameters
    __________
    column: pd.Series
        Column of the pd.DataFrame corresponding to 'SeriesDescription'.

    Returns
    _______
    pd.Series:
        Priority values for each of the original elements of 'column'.
    """
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
    """
    Call upon 'mr_series_selection' to add 'NormalizedSeriesDescription' and 'SeriesType'
    columns to a pd.DataFrame containing a single study.

    Parameters
    __________
    study_df: pd.DataFrame
        A DataFrame containing data for a single study. It must contain the following
        columns: ['SeriesDescription', 'dicoms'].
    ruleset: str
        Ruleset used within mr_series_selection to predict the NormalizedDescription of
        each series. Options include 'brain', 'lumbar', and 'prostate'. Defaults to 'brain'.
    description_key: dict
        Key for combining 'NormalizedDescription's defined by mr_series_selection into desired
        categories. This information is provided by using a path to a json file containing this
        information. If nothing is provided, the description_key will default to:
        default_key = {
            "T1Pre": [["iso3D AX T1 NonContrast", "iso3D AX T1 NonContrast RFMT"], "anat"],
            "T1Post": [["iso3D AX T1 WithContrast", "iso3D AX T1 WithContrast RFMT"], "anat"],
        }
    check_columns: bool
        Whether to check 'study_df' for the required columns. Defaults to True.

    Returns
    _______
    pd.DataFrame:
        A DataFrame that contains the new columns: 'NormalizedSeriesDescription' and 'SeriesType'
        if a series can be predicted by 'mr_series_selection'.
    """
    if check_columns:
        required_columns = ["SeriesDescription", "dicoms"]

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

        try:
            classification = get_series_classification(ruleset, dcms)
            normalized_description = classification.get("NormalizedDescription", None)
        except Exception:
            normalized_description = None

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

    return filtered_df


def series_from_csv(
    csv: Path | str,
    ruleset: str = "brain",
    description_key: dict = default_key,
    cpus: int = 1,
    check_columns: bool = True,
) -> pd.DataFrame:
    """
    Call upon 'mr_series_selection' to add 'NormalizedSeriesDescription' and 'SeriesType'
    columns to every study within a dataset.

    Parameters
    __________
    csv: Path | str
        The path to a CSV containing an entire dataset. It must contain the following
        columns: ['StudyInstanceUID', 'SeriesDescription', 'dicoms'].
    ruleset: str
        Ruleset used within mr_series_selection to predict the NormalizedDescription of
        each series. Options include 'brain', 'lumbar', and 'prostate'. Defaults to 'brain'.
    description_key: dict
        Key for combining 'NormalizedDescription's defined by mr_series_selection into desired
        categories. If nothing is provided, the description_key will default to:
        default_key = {
            "T1Pre": [["iso3D AX T1 NonContrast", "iso3D AX T1 NonContrast RFMT"], "anat"],
            "T1Post": [["iso3D AX T1 WithContrast", "iso3D AX T1 WithContrast RFMT"], "anat"],
        }
    cpus: int
        Number of cpus to use for multiprocessing. Defaults to 0 (no multiprocessing).
    check_columns: bool
        Whether to check the CSV for the required columns. Defaults to True.

    Returns
    _______
    pd.DataFrame:
        A DataFrame that contains the new columns: 'NormalizedSeriesDescription' and 'SeriesType'
        if a series can be predicted by 'mr_series_selection'. This function will also overwrite
        the input CSV with this DataFrame.
    """

    df = pd.read_csv(csv, dtype=str)

    if check_columns:
        required_columns = ["StudyInstanceUID", "SeriesDescription", "dicoms"]

        check_required_columns(df, required_columns)

    filtered_df = df.copy().dropna(subset="StudyInstanceUID")
    study_uids = filtered_df["StudyInstanceUID"].unique()

    kwargs_list = [
        {
            "study_df": filtered_df[
                filtered_df["StudyInstanceUID"] == study_uid
            ].copy(),
            "ruleset": ruleset,
            "description_key": description_key,
            "check_columns": False,
        }
        for study_uid in study_uids
    ]

    with tqdm(
        total=len(kwargs_list), desc="Predicting on studies"
    ) as pbar, ThreadPoolExecutor(cpus if cpus >= 1 else 1) as executor:
        futures = [executor.submit(series_in_study, **kwargs) for kwargs in kwargs_list]
        for future in as_completed(futures):
            prediction_df = future.result()
            df = (
                pd.read_csv(csv, dtype=str)
                .drop_duplicates(subset="SeriesInstanceUID")
                .reset_index(drop=True)
            )
            df = pd.merge(df, prediction_df, how="outer")
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
    df.to_csv(csv, index=False)
    return df


__all__ = [
    "default_key",
    "series_in_study",
    "series_from_csv",
]
