"""
The `slurm_concurrency` module serves as the entrypoint for the `spreprocessing` CLI. It functions
similarly to the standard `preprocessing` CLI but is designed for use with slurm. It has no public
functions intended for use in the Python API. Run 'spreprocessing -h' in the terminal for a command
usage guide.
"""
import pandas as pd
import os
import sys
import argparse

from pathlib import Path
from typing import Literal, Sequence, Dict, Any
from tqdm import tqdm
from subprocess import run


def generate_array_template(
    command: str,
    slurm_dir: Path | str,
    csv: Path | str,
    account: str = "qtim",
    partition: str = "basic",
    time: str = "05:00:00",
    memory: str = "10G",
    mail_update: bool = False,
    patients: Sequence[str] | None = None,
    cpus: int = 50,
    dependency: str | None = None,
) -> str:
    """
    Generate a script that can be used to launch an array of slurm jobs with `sbatch`.

    Parameters
    __________
    command: str
        The command you wish to run through the preprocessing cli, complete with its own flags.
        Do not include the flags associated with concurrency within the command itself. Only the
        `brain-preprocessing` is currently supported.

    slurm_dir: Path | str
        The directory that will contain the job array script and its outputs when this script is
        run.

    account: str
        The account to use for the slurm array. Defaults to 'qtim'.

    partition: str
        The partition to use for the slurm array. Defaults to 'basic'.

    time: str
        The maximum time allotted for each job within the slurm array. Defaults to '05:00:00'.

    memory: str
        The memory allotted for each job within the slurm array. Defaults to '10G'.

    mail_update: bool
        Whether to send a mail update upon the completion or failure of the slurm array. Defaults
        to `False`.

    patients: Sequence[str] | None
        A sequence of patients to select from the 'AnonPatientID' column of the CSV referenced in
        `command`. If 'None' is provided, all patients will be preprocessed.

    cpus: int
        The number of concurrent jobs to run within the slurm array. Defaults to 50.

    Returns
    _______
    Path:
        The path to the generated array script. 
    """ 
    if patients is None:
        df = pd.read_csv(csv, dtype=str)
        patients = list(df["AnonPatientID"].unique())
        num_patients = len(patients)

    else:
        num_patients = len(patients)

    slurm_dir = Path(slurm_dir).resolve()
    slurm_dir.mkdir(parents=True, exist_ok=True)

    script = "#!/bin/bash"
    
    if cpus > 1:
        script += f"\n#SBATCH --array={0}-{num_patients-1}%{min(cpus, num_patients)}"
        script += f"\n#SBATCH --output={slurm_dir}/%A_%a.out"
    else:
        script += f"\n#SBATCH --output={slurm_dir}/%j.out"
    
    script += f"\n#SBATCH --account={account}" if account != "" else ""
    script += f"\n#SBATCH --partition={partition}" if partition != "" else ""
    script += f"\n#SBATCH --time={time}" if time != "" else ""
    script += f"\n#SBATCH --mem-per-cpu={memory}" if memory != "" else ""

    if isinstance(dependency, str):
        script += f"\n#SBATCH --dependency=afterany:{dependency}"

    if mail_update:
        script += "\n#SBATCH --mail-type=END,FAIL"
 
    script += (
        f"\n\nexport PYENV_VERSION={os.environ['PYENV_VERSION']}"
        if "PYENV_VERSION" in os.environ
        else ""
    )
    script += f"\nexport SLURM_ARRAY_OUTPUTS={slurm_dir}"

    script += (f"\n\n{command}")

    if cpus > 1:
        outfile = slurm_dir / "primary_job.sh"

    else:
        outfile = slurm_dir / "aggregation_job.sh"

    with open(outfile, "w") as f:
        f.writelines(script)

    return str(outfile)


def aggregate_slurm_results(slurm_dir: Path | str, csv: Path | str) -> None:
    """
    Update the primary CSV with the results from the slurm array jobs. Do not run on directly
    on the cluster's login node.
    
    Parameters
    __________
    slurm_dir: Path | str
        The directory that containing the job array script and its outputs.

    csv: Path | str
        The path to the primary CSV that you wish to update.

    Returns
    _______
    pd.DataFrame
        A DataFrame updated to reflect the results of the slurm array jobs.
    """
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
            .sort_values(["AnonPatientID", "AnonStudyID"])
            .reset_index(drop=True)
        )
        df.to_csv(csv, index=False)

    df = (
        pd.read_csv(csv, dtype=str)
        .drop_duplicates(subset="SeriesInstanceUID")
        .sort_values(["AnonPatientID", "AnonStudyID"])
        .reset_index(drop=True)
    )
    df.to_csv(csv, index=False)


def launch_slurm(
    function_name: Literal["brain_preprocessing"],
    function_kwargs: Dict[str, Any],
    slurm_dir: Path | str,
    csv: Path | str,
    account: str = "qtim",
    partition: str = "basic",
    time: str = "05:00:00",
    memory: str = "10G",
    mail_update: bool = False,
    patients: Sequence[str] | None = None,
    cpus: int = 50,
) -> None:
    """
    Launch a primary job array and aggregation job for a supported funciton using slurm.

    Parameters
    __________
    function_name: Literal['brain_preprocessing']
        The basename of the function that will be called.

    function_kwargs: Dict[str, Any]
        The kwargs that will be used in the command when it is called.

    slurm_dir: Path | str
        The directory that will contain the job array script and its outputs when this script is
        run.

    csv: Path | str
        The primary CSV associated with running the desired function.

    account: str
        The account to use for the slurm array. Defaults to 'qtim'.

    partition: str
        The partition to use for the slurm array. Defaults to 'basic'.

    time: str
        The maximum time allotted for each job within the slurm array. Defaults to '05:00:00'.

    memory: str
        The memory allotted for each job within the slurm array. Defaults to '10G'.

    mail_update: bool
        Whether to send a mail update upon the completion or failure of the slurm array. Defaults
        to `False`.

    patients: Sequence[str] | None
        A sequence of patients to select from the 'AnonPatientID' column of the CSV referenced in
        `command`. If 'None' is provided, all patients will be preprocessed.

    cpus: int
        The number of concurrent jobs to run within the slurm array. Defaults to 50.

    Returns
    _______
    None
    """
    def quotewrap(s):
        return f"\"{s}\""

    slurm_dir = Path(slurm_dir).resolve()

    if function_name == "brain_preprocessing":
        primary_command = (
            "python -c \'from preprocessing.brain import preprocess_from_csv; "
            f"preprocess_from_csv({', '.join([f'{k}={quotewrap(v) if isinstance(v, (str, Path)) else v}' for k,v in function_kwargs.items()])})\'"
        )

    primary_jobfile = generate_array_template(
        command=primary_command,
        slurm_dir=slurm_dir,
        csv=csv,
        account=account,
        partition=partition,
        time=time,
        memory=memory,
        mail_update=mail_update,
        patients=patients,
        cpus=cpus,
    )

    primary_job = run(["sbatch", primary_jobfile], capture_output=True)
    primary_job_id = primary_job.stdout.strip().split()[-1].decode()

    aggregation_command = (
        "python -c 'from preprocessing.slurm_concurrency import aggregate_slurm_results; "
        f"aggregate_slurm_results({quotewrap(slurm_dir)}, {quotewrap(csv)})'"
    )

    aggregation_jobfile = generate_array_template(
        command=aggregation_command,
        slurm_dir=slurm_dir,
        csv=csv,
        account=account,
        partition=partition,
        time=time,
        memory=memory,
        mail_update=mail_update,
        patients=patients,
        cpus=1,
        dependency=primary_job_id
    )

    aggregation_job = run(["sbatch", aggregation_jobfile], capture_output=True)
    aggregation_job_id = aggregation_job.stdout.strip().split()[-1].decode()

    print(f"Submitted primary job: {primary_job_id} followed by aggregation job: {aggregation_job_id}")


USAGE_STR = """
spreprocessing [<slurm-args>] <command> [<command-args>]

This is the alternative slurm CLI for achieving concurrency on a compute cluster. Only use on
machines that support slurm. For normal use, return to the standard `preprocessing` CLI.

The following flag is required for all commands:
    --slurm-dir                 The directory that will contain the slurm jobscripts and their
                                outputs.

The following flags are optional for all commands:
    --account                   The account to use for the slurm array. Defaults to 'qtim'.

    --partition                 The partition to use for the slurm array. Defaults to 'basic'.

    --time                      The maximum time allotted for each job within the slurm array.
                                Defaults to '05:00:00'.

    --mem-per-cpu               The memory allotted for each job within the slurm array.
                                Defaults to '10G'.

    --mail-update               Whether to send a mail update upon the completion or failure of
                                the slurm array and aggregation job.

    -c | --cpus                 The number of concurrent jobs to run within the slurm array.
                                Defaults to 50.


The following commands are available: 
    brain-preprocessing         Preprocess NIfTI files for deep learning. A CSV is required to
                                indicate the location of source files and to procide the context
                                for filenames. The outputs will follow a BIDS inspired convention. 

Run `spreprocessing <command> --help` for more details about how to use each individual command.

"""

parser = argparse.ArgumentParser(usage=USAGE_STR)

parser.add_argument(
    "--slurm-dir",
    required=True,
    type=Path,
    help="The directory that will contain the slurm jobscripts and their outputs.",
)

parser.add_argument(
    "--account",
    type=str,
    default="qtim",
    help="The account to use for the slurm array. Defaults to 'qtim'.",
)

parser.add_argument(
    "--partition",
    type=str,
    default="basic",
    help="The partition to use for the slurm array. Defaults to 'basic'.",
)

parser.add_argument(
    "--time",
    type=str,
    default="05:00:00",
    help="The maximum time allotted for each job within the slurm array. Defaults to '05:00:00'.",
)

parser.add_argument(
    "--mem-per-cpu",
    type=str,
    default="10G",
    help="The memory allotted for each job within the slurm array. Defaults to '10G'.",
)

parser.add_argument(
    "--mail-update",
    action="store_true",
    help=(
        "Whether to send a mail update upon the completion or failure of the slurm "
        "array and aggregation job."
    ),
)

parser.add_argument(
    "-c",
    "--cpus",
    type=int,
    default=50,
    help=(
        "The number of concurrent jobs to run within the slurm array. Defaults to 50."
    ),
)

subparsers = parser.add_subparsers(dest="command")

brain_preprocessing = subparsers.add_parser(
    "brain-preprocessing",
    description=(
        """
        Preprocess NIfTI files for deep learning. A CSV is required to
        indicate the location of source files and to procide the context
        for filenames. The outputs will follow a BIDS inspired convention.
        """
    ),
)

brain_preprocessing.add_argument(
    "preprocessed_dir",
    metavar="preprocessed-dir",
    type=Path,
    help=("The directory that will contain the preprocessed NIfTI files."),
)

brain_preprocessing.add_argument(
    "csv",
    type=Path,
    help=(
        """
        A CSV containing NIfTI location and information required for the output file names.
        It must contain the columns: 'Nifti', 'AnonPatientID', 'AnonStudyID', 
        'StudyInstanceUID', 'SeriesInstanceUID', 'NormalizedSeriesDescription', and 'SeriesType'.
        """
    ),
)

brain_preprocessing.add_argument(
    "-p",
    "--patients",
    type=str,
    default=None,
    help=(
        """
        A comma delimited list of patients to select from the 'AnonPatientID' column
        of the CSV
        """
    ),
)

brain_preprocessing.add_argument(
    "-pk",
    "--pipeline-key",
    type=str,
    default="preprocessed",
    help=(
        """
        The key that will be used in the CSV to indicate the new locations of preprocessed
        files. Defaults to 'preprocessed'.
        """
    ),
)

brain_preprocessing.add_argument(
    "-rk",
    "--registration-key",
    type=str,
    default="T1Post",
    help=(
        """
        The value that will be used to select the fixed image during registration. This 
        should correspond to a value within the 'NormalizedSeriesDescription' column in
        the CSV. If you have segmentation files in your data. They should correspond to
        this same series. Defaults to 'T1Post'.
        """
    ),
)

brain_preprocessing.add_argument(
    "-l",
    "--longitudinal-registration",
    action="store_true",
    help=(
        """
        Whether to use longitudinal registration. Additional studies for the same patient
        will be registered to the first study (chronologically). False if not specified.
        """
    ),
)

brain_preprocessing.add_argument(
    "-a",
    "--atlas-target",
    type=Path,
    default=None,
    help=(
        """
        The path to an atlas file if using using an atlas for the registration. If provided,
        `--longitudinal-registration` will be ignored.
        """
    ),
)

brain_preprocessing.add_argument(
    "-m",
    "--model",
    choices=["rigid", "affine", "joint", "deform"],
    default="affine",
    help=(
        """
        The synthmorph model that will be used to perform registration. Choices are: 'rigid', 'affine', 'joint',
        and 'deform'. Defaults to 'affine'.
        """
    ),
)

brain_preprocessing.add_argument(
    "-o",
    "--orientation",
    type=str,
    default="RAS",
    help=(
        "The orientation standard that you wish to set for preprocessed data. Defaults to 'RAS'."
    ),
)

brain_preprocessing.add_argument(
    "-s",
    "--spacing",
    type=str,
    default="1,1,1",
    help=(
        """
        A comma delimited list indicating the desired spacing of preprocessed data. Measurements
        are in mm. Defaults to '1,1,1'.
        """
    ),
)

brain_preprocessing.add_argument(
    "-ns",
    "--no-skullstrip",
    action="store_true",
    help=(
        """
        Whether to not apply skullstripping to preprocessed data. Skullstripping will be
        applied if not specified.
        """
    ),
)

brain_preprocessing.add_argument(
    "-ps",
    "--pre-skullstripped",
    action="store_true",
    help=(
        """
        Whether the input data is already skullstripped. Skullstripping will not be applied
        if specified.
        """
    ),
)

brain_preprocessing.add_argument(
    "-b",
    "--binarize-seg",
    action="store_true",
    help=(
        """
        Whether to binarize segmentations. Not recommended for multi-class labels. Binarization is not
        applied by default.
        """
    ),
)

brain_preprocessing.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    help=(
        """
        If specified, the commands that are called and their outputs will be printed to
        the console.
        """
    ),
)


def slurm_cli() -> None:
    """
    The slurm CLI for the `preprocessing` library. Run 'spreprocessing -h' for additional help.
    """
    if len(sys.argv) == 1:
        parser.print_usage()
        exit(0)

    args = parser.parse_args()

    if args.command == "brain-preprocessing":

        if isinstance(args.patients, str):
            args.patients = args.patients.split(",")

        function_kwargs = {
            "csv": args.csv,
            "preprocessed_dir": args.preprocessed_dir,
            "pipeline_key": args.pipeline_key,
            "registration_key": args.registration_key,
            "longitudinal_registration": args.longitudinal_registration,
            "atlas_target": args.atlas_target,
            "registration_model": args.model,
            "orientation": args.orientation,
            "spacing": [float(s) for s in args.spacing.split(",")],
            "skullstrip": not args.no_skullstrip,
            "pre_skullstripped": args.pre_skullstripped,
            "binarize_seg": args.binarize_seg,
            "verbose": args.verbose
        }

        launch_slurm(
            function_name="brain_preprocessing",
            function_kwargs=function_kwargs,
            slurm_dir=args.slurm_dir,
            csv=args.csv,
            account=args.account,
            partition=args.partition,
            time=args.time,
            memory=args.mem_per_cpu,
            mail_update=args.mail_update,
            patients=args.patients,
            cpus=args.cpus,
        )

    exit(0)

__all__ = []
