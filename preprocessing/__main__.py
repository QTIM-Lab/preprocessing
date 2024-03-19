"""
This Module serves as the entrypoint for the `preprocessing` CLI. It has no public
functions intended for use in the Python API. Run 'preprocessing -h' in the terminal
for a command usage guide.
"""
import argparse
import json
import sys

from pathlib import Path
from typing import Callable, Dict, Any
from pycrumbs import tracked
from git import Repo, InvalidGitRepositoryError

try:
    Repo(__file__, search_parent_directories=True)
    disable_git_tracking = False

except InvalidGitRepositoryError:
    disable_git_tracking = True


@tracked(
    directory_parameter="record_dir",
    record_filename="preprocessing_cli_record.json",
    chain_records=True,
    disable_git_tracking=disable_git_tracking,
    seed_numpy=False,
    seed_tensorflow=False,
    seed_torch=False,
)
def tracked_command(
    func: Callable, kwargs: Dict[str, Any], record_dir: Path | str
) -> Any:
    """
    A standardized command format that runs a function from the `preprocessing` library
    and tracks useful information with pycrumbs.

    Parameters
    __________
    func: Callable
        The function that will be used for the command.

    kwargs: Dict[str, Any]
        The key-word arguments that will be passed to `func`.

    record_dir: Path | str
        The directory in which the pycrumbs tracking json will be stored (should be the
        same as the directory containing `func`'s outputs').

    Returns
    _______
    output: Any
        Whatever is returned by `func(**kwargs)`.
    """
    return func(**kwargs)


USAGE_STR = """
preprocessing <command> [<args>]

The following commands are available:
    old-project-anon-keys       Create anonymization keys for anonymous PatientID and VisitID
                                from previous QTIM organizational scheme. Should be compatible
                                with data following a following <Patient_ID>/<Study_ID> directory
                                hierarchy.

    nifti-dataset-anon-keys     Create anonymization keys for a dataset that starts within NIfTI
                                format. If the 'SeriesDescription's are not normalized,
                                'NormalizedSeriesDescription's must be obtained externally before
                                the NIfTI dataset can be reorganized.

    reorganize-dicoms           Reorganize DICOMs to follow the BIDS convention. Any DICOMs found
                                recursively within this directory will be reorganized (at least
                                one level of subdirectories is assumed). Anonomyzation keys for
                                PatientIDs and StudyIDs are provided within a CSV.

    reorganize-niftis           Reorganize a NIfTI dataset to follow BIDS convention. As NIfTI files
                                lack metadata, anonymization keys must be provided in the form of a
                                CSV, such as one obtained with `nifti-dataset-anon-keys`.

    dataset-to-nifti            Convert DICOMs to NIfTI file format. A CSV is required to map a
                                DICOM series to the resulting .nii.gz file and to provide
                                the context for filenames. The outputs will comply with the BIDS
                                conventions.

    predict-series              Predict the sequence type for every series in your dataset. A CSV
                                is required to indicate the location of the corresponding DICOMs.
                                Predictions are made using the mr_series_selection repo's analysis
                                of the DICOM header. A json can be provided to combine multiple
                                NormalizedDescriptions into a single category.

    brain-preprocessing         Preprocess NIfTI files for deep learning. A CSV is required to
                                indicate the location of source files and to procide the context
                                for filenames. The outputs will comply with BIDS conventions.

Run `preprocessing <command> --help` for more details about how to use each individual command.

"""


parser = argparse.ArgumentParser(usage=USAGE_STR)

subparsers = parser.add_subparsers(dest="command")

old_project_anon_keys = subparsers.add_parser(
    "old-project-anon-keys",
    description=(
        """
        Create anonymization keys for anonymous PatientID and StudyID
        from the previous QTIM organizational scheme. Is compatible 
        with data following a following <Patient_ID>/<Study_ID> directory
        hierarchy.
        """
    ),
)


old_project_anon_keys.add_argument(
    "input_dir",
    metavar="input-dir",
    type=str,
    help=(
        """
        The directory containing all of the dicom files for a project.
        It is expected to follow the <Patient_ID>/<Study_ID> convention.
        """
    ),
)

old_project_anon_keys.add_argument(
    "output_dir",
    metavar="output-dir",
    type=str,
    help=(
        """
        The directory that will contain the output CSV and potentially
        an error file.
        """
    ),
)

nifti_dataset_anon_keys = subparsers.add_parser(
    "nifti-dataset-anon-keys",
    description=(
        """
        Create anonymization keys for a dataset that starts within NIfTI
        format. If the 'SeriesDescription's are not normalized,
        'NormalizedSeriesDescription's must be obtained externally before
        the NIfTI dataset can be reorganized.
        """
    ),
)

nifti_dataset_anon_keys.add_argument(
    "nifti_dir",
    metavar="nifti-dir",
    type=Path,
    help=("The directory containing all of the NIfTI files you wish to anonymize."),
)

nifti_dataset_anon_keys.add_argument(
    "output_dir",
    metavar="output-dir",
    type=Path,
    help=(
        "The directory that will contain the output CSV and potentially an error file."
    ),
)

nifti_dataset_anon_keys.add_argument(
    "--normalized-descriptions",
    action="store_true",
    help=(
        "Whether the 'SeriesDescription' in the NIfTI file name is already normalized."
    ),
)

reorganize_d = subparsers.add_parser(
    "reorganize-dicoms",
    description=(
        """
        Reorganize DICOMs to follow the BIDS convention. Any DICOMs found
        recursively within this directory will be reorganized (at least
        one level of subdirectories is assumed). Anonomyzation keys for
        PatientIDs and StudyIDs are provided within a CSV.
        """
    ),
)

reorganize_d.add_argument(
    "original_dicom_dir",
    metavar="original-dicom-dir",
    type=Path,
    help=("The directory containing all of the DICOM files you wish to reorganize."),
)

reorganize_d.add_argument(
    "new_dicom_dir",
    metavar="new-dicom-dir",
    type=Path,
    help=(
        """
        The directory that will contain the same DICOM files reorganized to 
        follow the BIDS convention.
        """
    ),
)

reorganize_d.add_argument(
    "--anon-csv",
    type=Path,
    default=None,
    help=(
        """
        A CSV mapping PatientID and StudyInstanceUID to anonymous values. If None is
        provided, the anonymization will be inferred from the DICOM headers.
        """
    ),
)

reorganize_d.add_argument(
    "-c",
    "--cpus",
    type=int,
    default=1,
    help=(
        "Number of cpus to use for multiprocessing. Defaults to 1 (no multiprocessing)."
    ),
)

reorganize_n = subparsers.add_parser(
    "reorganize-niftis",
    description=(
        """
        Reorganize a NIfTI dataset to follow BIDS convention. As NIfTI files
        lack metadata, anonymization keys must be provided in the form of a
        CSV, such as one obtained with `nifti-dataset-anon-keys`.
        """
    ),
)

reorganize_n.add_argument(
    "nifti_dir",
    metavar="nifti-dir",
    type=Path,
    help=("The directory in which the reorganized NIfTIs will be stored."),
)

reorganize_n.add_argument(
    "anon_csv",
    metavar="anon-csv",
    type=Path,
    help=(
        """
        A CSV containing the original location of NIfTI files and metadata
        required for preprocessing commands. It must contain the columns: 
        'Anon_PatientID', 'Anon_StudyID', 'PatientID', 'StudyDate',
        'SeriesInstanceUID', 'StudyInstanceUID', 'SeriesDescription',
        'original_nifti', and 'NormalizedSeriesDescription'. 'SeriesType'
        can also be provided, otherwise "anat" will be assumed.
        """
    ),
)

reorganize_n.add_argument(
    "-c",
    "--cpus",
    type=int,
    default=1,
    help=(
        "Number of cpus to use for multiprocessing. Defaults to 1 (no multiprocessing)."
    ),
)


# def validate_bids(self):
#     paths = sys.argv[2:]
#
#     if ("--help" in paths) or ("-h" in paths):
#         print(
#             "File(s) validated for BIDS convention. Use spaces to delimit multiple files."
#         )
#     else:
#         validate(paths)


dataset_to_nifti = subparsers.add_parser(
    "dataset-to-nifti",
    description=(
        """
        Convert DICOMs to NIfTI file format. A CSV is required to map a
        DICOM series to the resulting .nii.gz file and to provide
        the context for filenames. The outputs will comply with the BIDS
        conventions.
        """
    ),
)

dataset_to_nifti.add_argument(
    "nifti_dir",
    metavar="nifti-dir",
    type=Path,
    help=("The directory that will contain the converted NIfTI files."),
)

dataset_to_nifti.add_argument(
    "csv",
    type=Path,
    help=(
        """
        A CSV containing dicom location and information required for the nifti file
        names. It must contain the columns: ['dicoms', 'Anon_PatientID', 
        'Anon_StudyID', 'StudyInstanceUID', 'SeriesInstanceUID', 'Manufacturer', 'NormalizedSeriesDescription',
        'SeriesType'].
        """
    ),
)

dataset_to_nifti.add_argument(
    "--overwrite",
    action="store_true",
    help="Whether to overwrite the .nii.gz files. False if not specified.",
)

dataset_to_nifti.add_argument(
    "-c",
    "--cpus",
    type=int,
    default=1,
    help=(
        "Number of cpus to use for multiprocessing. Defaults to 1 (no multiprocessing)."
    ),
)

predict_series = subparsers.add_parser(
    "predict-series",
    description=(
        """
        Predict the sequence type for every series in your dataset. A CSV
        is required to indicate the location of the corresponding DICOMs.
        Predictions are made using the mr_series_selection repo's analysis
        of the DICOM header. A json can be provided to combine multiple
        NormalizedDescriptions into a single category.
        """
    ),
)

predict_series.add_argument(
    "csv",
    type=Path,
    help=(
        """
        The path to a CSV containing an entire dataset. It must contain the following
        columns: ['StudyInstanceUID', 'SeriesInstanceUID', 'SeriesDescription', 'dicoms'].
        """
    ),
)

predict_series.add_argument(
    "--ruleset",
    type=str,
    default="brain",
    help=(
        """
        Ruleset used within mr_series_selection to predict the NormalizedDescription of
        each series. Options include 'brain', 'lumbar', and 'prostate'. Defaults to 'brain'.
        """
    ),
)

predict_series.add_argument(
    "--description-key",
    type=str,
    default=None,
    help=(
        """
        Key for combining 'NormalizedDescription's defined by mr_series_selection into desired
        categories. This information is provided by using a path to a json file containing this
        information. If nothing is provided, the description_key will default to: 

        DEFAULT_SERIES_KEY = {
            "T1Pre": [["iso3D AX T1 NonContrast", "iso3D AX T1 NonContrast RFMT"], "anat"],
            "T1Post": [["iso3D AX T1 WithContrast", "iso3D AX T1 WithContrast RFMT"], "anat"],
        }
        """
    ),
)

predict_series.add_argument(
    "-c",
    "--cpus",
    type=int,
    default=1,
    help=(
        "Number of cpus to use for multiprocessing. Defaults to 1 (no multiprocessing)."
    ),
)

brain_preprocessing = subparsers.add_parser(
    "brain-preprocessing",
    description=(
        """
        Preprocess NIfTI files for deep learning. A CSV is required to
        indicate the location of source files and to procide the context
        for filenames. The outputs will comply with BIDS conventions.
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
        A CSV containing nifti location and information required for the output file names.
        It must contain the columns: 'nifti', 'Anon_PatientID', 'Anon_StudyID', 
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
        A comma delimited list of patients to select from the 'Anon_PatientID' column
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
    "-c",
    "--cpus",
    type=int,
    default=1,
    help=(
        "Number of cpus to use for multiprocessing. Defaults to 1 (no multiprocessing)."
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

brain_preprocessing.add_argument(
    "-d",
    "--debug",
    action="store_true",
    help=(
        """
        Whether to run in debug mode. Each intermediate step will be saved using a suffix
        for differentiation. The input CSV will not be altered. Instead, a new copy will
        be saved to the output directory.
        """
    ),
)


def main() -> None:
    """
    The CLI for the `preprocessing` library. Run 'preprocessing -h' for additional help.
    """
    if len(sys.argv) == 1:
        parser.print_usage()
        exit(0)

    args = parser.parse_args()

    if args.command == "old-project-anon-keys":
        from preprocessing.bids import find_anon_keys

        kwargs = {"input_dir": args.input_dir, "output_dir": args.output_dir}

        tracked_command(find_anon_keys, kwargs=kwargs, record_dir=args.output_dir)

    elif args.command == "nifti-dataset-anon-keys":
        from preprocessing.bids import nifti_anon_csv

        kwargs = {
            "nifti_dir": args.nifti_dir,
            "output_dir": args.output_dir,
            "normalized_descriptions": args.normalized_descriptions,
        }

        tracked_command(nifti_anon_csv, kwargs=kwargs, record_dir=args.output_dir)

    elif args.command == "reorganize-dicoms":
        from preprocessing.bids import reorganize_dicoms

        kwargs = {
            "original_dicom_dir": args.original_dicom_dir,
            "new_dicom_dir": args.new_dicom_dir,
            "anon_csv": args.anon_csv,
            "cpus": args.cpus,
        }

        tracked_command(reorganize_dicoms, kwargs=kwargs, record_dir=args.new_dicom_dir)

    elif args.command == "reorganize-niftis":
        from preprocessing.bids import reorganize_niftis

        kwargs = {
            "nifti_dir": args.nifti_dir,
            "anon_csv": args.anon_csv,
            "cpus": args.cpus,
        }

        tracked_command(reorganize_niftis, kwargs=kwargs, record_dir=args.nifti_dir)

    elif args.command == "dataset-to-nifti":
        from preprocessing.bids import convert_batch_to_nifti

        kwargs = {
            "nifti_dir": args.nifti_dir,
            "csv": args.csv,
            "overwrite_nifti": args.overwrite,
        }

        tracked_command(
            convert_batch_to_nifti, kwargs=kwargs, record_dir=args.nifti_dir
        )

    elif args.command == "predict-series":
        from preprocessing.series_selection import series_from_csv

        kwargs = {
            "csv": args.csv,
            "ruleset": args.ruleset,
            "cpus": args.cpus,
        }

        if args.description_key is not None:
            kwargs["description_key"] = json.load(open(args.description_key, "r"))

        tracked_command(series_from_csv, kwargs=kwargs, record_dir=args.csv.parent)

    elif args.command == "brain-preprocessing":
        from preprocessing.brain import preprocess_from_csv

        if isinstance(args.patients, str):
            args.patients = args.patients.split(",")

        kwargs = {
            "csv": args.csv,
            "preprocessed_dir": args.preprocessed_dir,
            "patients": args.patients,
            "pipeline_key": args.pipeline_key,
            "registration_key": args.registration_key,
            "longitudinal_registration": args.longitudinal_registration,
            "registration_model": args.model,
            "orientation": args.orientation,
            "spacing": [float(s) for s in args.spacing.split(",")],
            "skullstrip": not args.no_skullstrip,
            "pre_skullstripped": args.pre_skullstripped,
            "binarize_seg": args.binarize_seg,
            "cpus": args.cpus,
            "verbose": args.verbose,
            "debug": args.debug,
        }

        tracked_command(
            preprocess_from_csv, kwargs=kwargs, record_dir=args.preprocessed_dir
        )

    exit(0)


__all__ = []
