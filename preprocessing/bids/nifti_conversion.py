import multiprocessing
import os
import pandas as pd

from subprocess import run
from typing import Literal
from tqdm import tqdm
from pathlib import Path


def convert_to_nifti(
    dicom_dir: Path | str,
    nifti_dir: Path | str,
    anon_patientID: str,
    anon_studyID: str,
    normalized_series_description: str,
    subdir: Literal["anat", "func", "dwi"],
    overwrite: bool = False,
) -> str | None:
    if isinstance(nifti_dir, str):
        nifti_dir = Path(nifti_dir)

    output_dir = nifti_dir / anon_patientID / anon_studyID / subdir
    output_nifti = (
        output_dir
        / f"{anon_patientID}_{anon_studyID}_{normalized_series_description.replace(' ', '_')}"
    )

    if (not overwrite) and output_nifti.with_suffix(".nii.gz").exists():
        return str(output_nifti) + ".nii.gz"

    os.makedirs(output_dir, exist_ok=True)

    command = f"dcm2niix -z y -f {output_nifti.name} -o {output_dir} -b y {dicom_dir}"
    print(command)

    run(
        command.split(" "),
        env={"PATH": "/usr/pubsw/packages/fsl/6.0.6/bin/:" + os.environ["PATH"]},
    )

    return str(output_nifti) + ".nii.gz"


def convert_study(
    study_df: pd.DataFrame, nifti_dir: Path, overwrite_nifti: bool = False
) -> pd.DataFrame:
    """
    Helper function for convert_batch_to_nifti
    """
    series_descriptions = study_df["NormalizedSeriesDescription"].unique()

    series_dfs = []
    for series_description in series_descriptions:
        series_df = study_df[
            study_df["NormalizedSeriesDescription"] == series_description
        ]
        output_niftis = []

        for i in range(series_df.shape[0]):
            output_nifti = convert_to_nifti(
                dicom_dir=series_df.loc[series_df.index[i], "dicoms"],
                nifti_dir=nifti_dir,
                anon_patientID=series_df.loc[series_df.index[i], "Anon_PatientID"],
                anon_studyID=series_df.loc[series_df.index[i], "Anon_StudyID"],
                normalized_series_description=series_description,
                subdir=series_df.loc[series_df.index[i], "SeriesType"],
                overwrite=overwrite_nifti,
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
    df = pd.read_csv(csv)

    filtered_df = df.drop_na(subset="NormalizedSeriesDescription")

    study_uids = filtered_df["StudyInstanceUID"].unique()

    if cpus == 0:
        outputs = [
            convert_study(
                filtered_df[filtered_df["StudyInstanceUID"] == study_uid],
                nifti_dir,
                overwrite_nifti,
            )
            for study_uid in tqdm(study_uids, desc="Converting to NIfTI")
        ]

    else:
        inputs = [
            [
                filtered_df[filtered_df["StudyInstanceUID"] == study_uid],
                nifti_dir,
                overwrite_nifti,
            ]
            for study_uid in study_uids
        ]

        with multiprocessing.Pool(cpus) as pool:
            outputs = list(
                tqdm(
                    pool.imap(convert_study_star, inputs),
                    total=len(study_uids),
                    desc="Converting to  NIfTI",
                )
            )

    for output in outputs:
        df = df.update(output)

    df.to_csv(csv, index=False)
