import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

from pathlib import Path
from SimpleITK import ReadImage, GetArrayFromImage
from tqdm import tqdm
from typing import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from preprocessing.utils import check_required_columns


def vol_plot_patient(
    patient_df: pd.DataFrame, plot_dir: Path | str, pipeline_key: str = "preprocessed"
) -> pd.DataFrame:
    patient_id = patient_df.loc[patient_df.index[0], "Anon_PatientID"]

    keys = list(
        filter(
            lambda x: re.search(f"{pipeline_key}_label.+_ids$", x), patient_df.keys()
        )
    )

    rows = []
    for key in keys:
        label = key.replace(f"{pipeline_key}_label", "").replace("_ids", "")

        filtered_df = patient_df.dropna(subset=key)

        study_uids = patient_df["Anon_StudyID"]

        arrays = [GetArrayFromImage(ReadImage(file)) for file in filtered_df[key]]

        total_tumors = max([array.max() for array in arrays])

        dates = pd.to_datetime(filtered_df["StudyDate"], format="%Y%m%d")

        normalized_dates = [date.days / 365 for date in (dates - dates.min())]

        max_volumes = {}

        for array, date, study_uid in zip(arrays, normalized_dates, study_uids):
            for i in range(1, total_tumors + 1):
                volume = (array == i).astype(int).sum()
                max_volumes[i] = max(volume, max_volumes.get(i, 0))

                rows.append(
                    {
                        "Anon_PatientID": patient_id,
                        "Anon_StudyID": study_uid,
                        "Label": label,
                        "Tumor ID": f"{i:03d}",
                        "Volume": volume,
                        "Relative Date [Y]": date,
                    }
                )

        for i in range(len(rows)):
            tumor_id = int(rows[i]["Tumor ID"])
            volume = rows[i]["Volume"]
            rows[i]["Normalized Volume"] = volume / max_volumes[tumor_id]

    out_df = pd.DataFrame(rows)
    
    if out_df.empty:
        return out_df

    out_dir = Path(plot_dir).resolve() / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.lineplot(out_df, x="Relative Date [Y]", y="Normalized Volume", hue="Tumor ID")
    plt.tight_layout()
    plt.savefig(out_dir / "normalized_volume.png")
    plt.close()

    sns.lineplot(out_df, x="Relative Date [Y]", y="Volume", hue="Tumor ID")
    plt.tight_layout()
    plt.savefig(out_dir / "true_volume.png")
    plt.close()

    return out_df

def vol_plot_csv(
    csv: Path | str,
    plot_dir: Path | str,
    patients: Sequence[str] | None,
    pipeline_key: str = "preprocessed",
    cpus: int = 1,
) -> pd.DataFrame:
    """
    Assign tumor IDs in all of the segmentations masks for every patient
    within a dataset.

    Parameters
    __________
    csv: Path | str
        The path to a CSV containing an entire dataset. It must contain the following columns:
        'Anon_PatientID', 'Anon_StudyID', 'SeriesType', and f'{pipeline_key}_seg'. Additionally,
        the previous preprocessing is assumed to have been registered longitudinally or to an
        atlas.

    tracking_dir: Path | str
        The directory that will contain the tumor id mask files.

    patients: Sequence[str] | None
        A sequence of patients to select from the 'Anon_PatientID' column of the CSV. If 'None'
        is provided, all patients will be preprocessed.

    pipeline_key: str
        The key used in the CSV when preprocessing was performed. Defaults to 'preprocessed'.

    labels: Sequence[int]
        A sequence of the labels included in the segmentation masks.

    cpus: int
        Number of cpus to use for multiprocessing. Defaults to 1 (no multiprocessing).

    Returns
    _______
    pd.DataFrame
        A Dataframe with added columns f'{pipeline_key}_label{label}_ids' for each provided
        label. This function will also overwrite the input CSV with this DataFrame.
    """
    df = pd.read_csv(csv, dtype=str)

    required_columns = [
        "Anon_PatientID",
        "Anon_StudyID",
        "SeriesType",
        f"{pipeline_key}_seg"
    ]

    check_required_columns(df, required_columns)

    plot_dir = Path(plot_dir).resolve()
    summary_csv = plot_dir / "summary.csv"

    if summary_csv.exists():
        os.remove(summary_csv)

    filtered_df = df.dropna(subset=[f"{pipeline_key}_seg"])

    if patients is None:
        patients = list(filtered_df["Anon_PatientID"].unique())

    kwargs_list = [
        {
            "patient_df": filtered_df[filtered_df["Anon_PatientID"] == patient].copy(),
            "plot_dir": plot_dir,
            "pipeline_key": pipeline_key,
        }
        for patient in patients
    ]

    with tqdm(
        total=len(kwargs_list), desc="Plotting Volumetric Change"
    ) as pbar, ProcessPoolExecutor(cpus if cpus >= 1 else 1) as executor:
        futures = [
            executor.submit(vol_plot_patient, **kwargs) for kwargs in kwargs_list
        ]
        for future in as_completed(futures):
            plotted_df = future.result()
            if plotted_df.empty:
                pbar.update(1)
                continue

            if summary_csv.exists():
                df = (
                    pd.read_csv(summary_csv, dtype=str)
                    .drop_duplicates(
                        subset=[
                            "Anon_PatientID",
                            "Label",
                            "Tumor ID",
                            "Relative Date [Y]",
                        ]
                    )
                    .reset_index(drop=True)
                )
                df = pd.concat((df, plotted_df))

            else:
                df = plotted_df

            df = (
                df 
                .sort_values(["Anon_PatientID", "Label", "Tumor ID", "Relative Date [Y]"])
                .reset_index(drop=True)
            )
            df.to_csv(summary_csv, index=False)
            pbar.update(1)

    df = (
        pd.read_csv(summary_csv, dtype=str)
        .drop_duplicates(
            subset=[
                "Anon_PatientID",
                "Label",
                "Tumor ID",
                "Relative Date [Y]",
            ]
        )

        .sort_values(["Anon_PatientID", "Label", "Tumor ID", "Relative Date [Y]"])
        .reset_index(drop=True)
    )
    df.to_csv(summary_csv, index=False)

    return df

__all__ = ["vol_plot_patient", "vol_plot_csv"]
