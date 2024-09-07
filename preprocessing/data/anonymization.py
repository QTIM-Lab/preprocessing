import pandas as pd

from preprocessing.utils import check_required_columns


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

