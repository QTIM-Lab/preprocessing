import pandas as pd
import os

from pathlib import Path
from typing import Sequence
from tqdm import tqdm


def generate_array_template(
    command: str,
    slurm_dir: Path | str,
    account: str = "qtim",
    partition: str = "basic",
    time: str = "05:00:00",
    memory: str = "10G",
    mail_update: bool = False,
    patients: Sequence[str] | None = None,
    cpus: int = 50,
) -> None:
    assert not any(
        [arg in command.split(" ") for arg in ["-p", "--patients", "-c", "--cpus"]]
    ), (
        "The -p/--patients and -c/--cpus arguments should not appear in the base command itself. "
        "Aborting template generation."
    )

    supported_commands = ["brain-preprocessing"]

    base_command = command.split(" ")[0]

    assert base_command in supported_commands, (
        f"{base_command} is not supported for slurm array jobs. "
        "Aborting template generation."
    )

    if patients is None:
        csv = [arg for arg in command.split(" ") if ".csv" in arg][0]

        df = pd.read_csv(csv, dtype=str)

        patients = list(df["Anon_PatientID"].unique())

    num_patients = len(patients)

    slurm_dir = Path(slurm_dir).resolve()
    slurm_dir.mkdir(parents=True, exist_ok=True)

    script = "#!/bin/bash"
    script += f"\n#SBATCH --array={0}-{num_patients-1}%{cpus}"
    script += f"\n#SBATCH --output={slurm_dir.resolve()}/%A_%a.out"
    script += f"\n#SBATCH --account={account}" if account != "" else ""
    script += f"\n#SBATCH --partition={partition}" if partition != "" else ""
    script += f"\n#SBATCH --time={time}" if time != "" else ""
    script += f"\n#SBATCH --mem-per-cpu={memory}" if memory != "" else ""

    if mail_update:
        script += "\n#SBATCH --mail-type=END,FAIL"

    script += "\n\ndeclare -A patients"
    for i, patient in enumerate(patients):
        script += f"\npatients[{i}]={patient}"

    script += (
        f"\n\nexport PYENV_VERSION={os.environ['PYENV_VERSION']}"
        if "PYENV_VERSION" in os.environ
        else ""
    )
    script += f"\nexport SLURM_ARRAY_OUTPUTS={slurm_dir}"

    script += (
        f"\n\npreprocessing {command}"
        + " --patients ${patients[${SLURM_ARRAY_TASK_ID}]}"
    )

    with open(slurm_dir / f"{base_command}.sh", "w") as f:
        f.writelines(script)


def aggregate_slurm_results(slurm_dir: Path | str, csv: Path | str) -> pd.DataFrame:
    slurm_dir = Path(slurm_dir).resolve()

    slurm_csvs = list(slurm_dir.glob("**/*.csv"))

    df = pd.read_csv(csv, dtype=str)

    for slurm_csv in tqdm(slurm_csvs, desc="Aggregating slurm results"):
        slurm_df = pd.read_csv(slurm_csv, dtype=str)

        df = (
            pd.read_csv(csv, dtype=str)
            .drop_duplicates(subset="SeriesInstanceUID")
            .reset_index(drop=True)
        )
        df = pd.merge(df, slurm_df, how="outer")
        df = (
            df.drop_duplicates(subset="SeriesInstanceUID")
            .sort_values(["Anon_PatientID", "Anon_StudyID"])
            .reset_index(drop=True)
        )
        df.to_csv(csv, index=False)

    df = (
        pd.read_csv(csv, dtype=str)
        .drop_duplicates(subset="SeriesInstanceUID")
        .sort_values(["Anon_PatientID", "Anon_StudyID"])
        .reset_index(drop=True)
    )
    df.to_csv(csv, index=False)
    return df
