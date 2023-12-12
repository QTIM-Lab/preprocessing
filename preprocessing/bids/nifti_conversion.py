import multiprocessing
import os
import pandas as pd

from subprocess import run
from typing import Literal
from tqdm import tqdm
from pathlib import Path
from preprocessing.utils import source_external_software, check_required_columns
from dicom2nifti import convert_directory


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
    if source_software:
        source_external_software()

    nifti_dir = Path(nifti_dir)

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

    os.makedirs(output_dir, exist_ok=True)

    command = f"dcm2niix -z y -f {output_nifti.name} -o {output_dir} -b y -w {int(overwrite)} {dicom_dir}"
    print(command)

    run(command.split(" "))

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
    Helper function for convert_batch_to_nifti
    """
    if source_software:
        source_external_software()

    if check_columns:
        required_columns = [
            "dicoms",
            "Anon_PatientID",
            "Anon_StudyID",
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
                dicom_dir=series_df.loc[series_df.index[i], "dicoms"],
                nifti_dir=nifti_dir,
                anon_patient_id=series_df.loc[series_df.index[i], "Anon_PatientID"],
                anon_study_id=series_df.loc[series_df.index[i], "Anon_StudyID"],
                manufacturer=series_df.loc[series_df.index[i], "Manufacturer"],
                normalized_series_description=series_description,
                subdir=series_df.loc[series_df.index[i], "SeriesType"],
                overwrite=overwrite_nifti,
                source_software=False,
            )
            output_niftis.append(output_nifti)

        series_df["nifti"] = output_niftis
        series_dfs.append(series_df)

    df = pd.concat(series_dfs)

    return df


def convert_study_star(args):
    return convert_study(*args)


def convert_batch_to_nifti(
    nifti_dir: Path | str,
    csv: Path | str,
    overwrite_nifti: bool = False,
    cpus: int = 0,
) -> pd.DataFrame:
    source_external_software()

    df = pd.read_csv(csv)

    required_columns = [
        "dicoms",
        "Anon_PatientID",
        "Anon_StudyID",
        "StudyInstanceUID",
        "Manufacturer",
        "NormalizedSeriesDescription",
        "SeriesType",
    ]

    check_required_columns(df, required_columns)

    filtered_df = df.copy().dropna(subset="NormalizedSeriesDescription")

    study_uids = filtered_df["StudyInstanceUID"].unique()

    if cpus == 0:
        outputs = [
            convert_study(
                filtered_df[filtered_df["StudyInstanceUID"] == study_uid].copy(),
                nifti_dir,
                overwrite_nifti,
                False,
                False,
            )
            for study_uid in tqdm(study_uids, desc="Converting to NIfTI")
        ]

    else:
        inputs = [
            [
                filtered_df[filtered_df["StudyInstanceUID"] == study_uid].copy(),
                nifti_dir,
                overwrite_nifti,
                False,
                False,
            ]
            for study_uid in study_uids
        ]

        with multiprocessing.Pool(cpus) as pool:
            outputs = list(
                tqdm(
                    pool.imap(convert_study_star, inputs),
                    total=len(study_uids),
                    desc="Converting to NIfTI",
                )
            )

    nifti_df = pd.concat(outputs)
    df = pd.merge(df, nifti_df, how="outer")
    df = df.sort_values(["Anon_PatientID", "Anon_StudyID"]).reset_index(drop=True)
    df.to_csv(csv, index=False)
    return df
