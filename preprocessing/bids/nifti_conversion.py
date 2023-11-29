import multiprocaessing
import os
import pandas as pd

from subprocess import run
from pathlib import Path
from preprocessing.constants import (
    ANAT_SERIES_DESCRIPTIONS,
    FUNC_SERIES_DESCRIPTIONS,
    DWI_SERIES_DESCRIPTIONS,
    ALL_SERIES_DESCRIPTIONS,
)


def convert_nifti(
    dicom_dir: Path | str,
    nifti_dir: Path | str,
    anon_patientID: str,
    anon_studyID: str,
    normalized_series_description: str,
    overwrite: bool = False,
    suffix: str = "",
) -> str | None:
    if isinstance(nifti_dir, str):
        nifti_dir = Path(nifti_dir)

    if normalized_series_description in ANAT_SERIES_DESCRIPTIONS:
        subdir = "anat"
    elif normalized_series_description in FUNC_SERIES_DESCRIPTIONS:
        subdir = "func"
    elif normalized_series_description in DWI_SERIES_DESCRIPTIONS:
        subdir = "dwi"
    else:
        return None

    output_dir = nifti_dir / anon_patientID / anon_studyID / subdir
    output_nifti = (
        output_dir
        / f"{anon_patientID}-{anon_studyID}-{normalized_series_description.replace(' ', '_') + suffix}"
    )

    if (not overwrite) and output_nifti.with_suffix(".nii.gz").exists():
        return str(output_nifti) + ".nii.gz"

    os.makedirs(output_dir, exist_ok=True)

    command = f"dcm2niix {dicom_dir} -z y -f {output_nifti} -o {output_dir}"
    print(command)

    run(
        command.split(" "),
        env={"PATH": "/usr/pubsw/packages/fsl/6.0.6/bin/:" + os.environ["PATH"]},
    )

    return str(output_nift) + ".nii.gz"


def convert_study(study_df: pd.DataFrame, nifti_dir: Path, overwrite_nifti: bool) -> pd.DataFrame:
    """
    Helper function for convert_batch_to_nifti
    """
    series_descriptions = study_df["NormalizedSeriesDescription"].unique()

    series_dfs = []
    for series_description in series_descriptions:
        series_df = study_df[study_df["NormalizedSeriesDescription"] == series_description]
        output_niftis = []

        for i in range(df.shape[0]):
            output_nifti = convert_nifti(
                dicom_dir=series_df.loc[series_df.index[i], "dicoms"],
                nifti_dir=nifti_dir,
                anon_patientID: series_df.loc[series_df.index[i], "Anon_PatientID"],
                anon_studyID: series_df
                normalized_series_description=series_description,
                overwrite=overwrite_nifti,
                suffix=f"_{i}",
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

    filtered_df = df[df["NormalizedSeriesDescription"].str.isin(ALL_SERIES_DESCRIPTIONS)]

    study_uids = filtered_df["StudyInstanceUID"].unique()


    # if cpus == 0:
    #     outputs = [
    #         copy_dicoms(sub_dir, new_dicom_dir, anon_df)
    #         for sub_dir in tqdm(sub_dirs, desc="Copying DICOMs")
    #     ]
    #
    # else:
    #     inputs = [[sub_dir, new_dicom_dir, anon_df] for sub_dir in sub_dirs]
    #
    #     with multiprocessing.Pool(cpus) as pool:
    #         outputs = list(
    #             tqdm(
    #                 pool.imap(copy_dicoms_star, inputs),
    #                 total=len(sub_dirs),
    #                 desc="Copying DICOMs",
    #             )
    #         )
    #
    # df = (
    #     pd.concat(outputs, ignore_index=True)
    #     .drop_duplicates(subset=["SeriesInstanceUID"])
    #     .sort_values("Anon_PatientID")
    #     .reset_index(drop=True)
    # )
    # df.to_csv((new_dicom_dir / "reorganization.csv"), index=False)
    #
    # return df
