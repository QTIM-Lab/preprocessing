import argparse
import json

from pathlib import Path
from pycrumbs import tracked
from typing import Callable, Dict, Any

from preprocessing.bids import convert_batch_to_nifti
from preprocessing.bids import find_anon_keys
from preprocessing.bids import reorganize_dicoms
from preprocessing.bids import validate
from preprocessing.brain import preprocess_from_csv, debug_from_csv
from preprocessing.series_selection import series_from_csv, default_key


@tracked(
    directory_parameter="record_dir",
    record_filename="preprocessing_cli_record.json",
    chain_records=True,
)
def tracked_command(func: Callable, kwargs: Dict[str, Any], record_dir: Path | str):
    return func(**kwargs)


USAGE_STR = """
preprocessing <command> [<args>]

The following commands are available:
    old-project-anon-keys       Create anonymization keys for anonymous PatientID and VisitID
                                from previous QTIM organizational scheme. Should be compatible
                                with data following a following <Patient_ID>/<Study_ID> directory
                                hierarchy.

    reorganize-dicoms           Reorganize DICOMs to follow the BIDS convention. Any DICOMs found
                                recursively within this directory will be reorganized (at least
                                one level of subdirectories is assumed). Anonomyzation keys for
                                PatientIDs and StudyIDs are provided within a csv.

    dataset-to-nifti            Convert DICOMs to NIfTI file format. A csv is required to map a
                                DICOM series to the resulting .nii.gz file and to provide
                                the context for filenames. The outputs will comply with the BIDS
                                conventions.

    predict-series              Predict the sequence type for every series in your dataset. A csv
                                is required to indicate the location of the corresponding DICOMs.
                                Predictions are made using the mr_series_selection repo's analysis
                                of the DICOM header. A json can be provided to combine multiple
                                NormalizedDescriptions into a single category.

    brain-preprocessing         Preprocess NIfTI files for deep learning. A csv is required to
                                indicate the location of source files and to procide the context
                                for filenames. The outputs will comply with BIDS conventions.

    debug-preprocessing         A debugging counterpart to `brain-preprocessing`, in which each
                                step is differentiated by a suffix added to the file names.
                                Debugging can also be performed on specific patients.

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
        The directory that will contain the output csv and potentially
        an error file.
        """
    ),
)

reorganize = subparsers.add_parser(
    "reorganize-dicoms",
    description=(
        """
        Reorganize DICOMs to follow the BIDS convention. Any DICOMs found
        recursively within this directory will be reorganized (at least
        one level of subdirectories is assumed). Anonomyzation keys for
        PatientIDs and StudyIDs are provided within a csv.
        """
    ),
)

reorganize.add_argument(
    "original_dicom_dir",
    metavar="original-dicom-dir",
    type=Path,
    help=("The directory containing all of the DICOM files you wish to reorganize."),
)

reorganize.add_argument(
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

reorganize.add_argument(
    "--anon-csv",
    type=Path,
    default=None,
    help=(
        """
        A csv mapping PatientID and StudyInstanceUID to anonymous values. If None is
        provided, the anonymization will be inferred from the DICOM headers.
        """
    ),
)

reorganize.add_argument(
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
        Convert DICOMs to NIfTI file format. A csv is required to map a
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
        A csv containing dicom location and information required for the nifti file
        names. It must contain the columns: ['dicoms', 'Anon_PatientID', 
        'Anon_StudyID', 'StudyInstanceUID', 'Manufacturer', 'NormalizedSeriesDescription',
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
        Predict the sequence type for every series in your dataset. A csv
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
        columns: ['StudyInstanceUID', 'SeriesDescription', 'dicoms'].
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

        default_key = {
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
        Preprocess NIfTI files for deep learning. A csv is required to
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
        A csv containing nifti location and information required for the output file names.
        It must contain the columns: 'nifti', 'Anon_PatientID', 'Anon_StudyID', 
        'StudyInstanceUID', 'NormalizedSeriesDescription', and 'SeriesType'.
        """
    ),
)

brain_preprocessing.add_argument(
    "--pipeline-key",
    type=str,
    default="preprocessed",
    help=(
        """
        The key that will be used in the csv to indicate the new locations of preprocessed 
        files. Defaults to 'preprocessed'.
        """
    ),
)

brain_preprocessing.add_argument(
    "--registration-key",
    type=str,
    default="T1Post",
    help=(
        """
        The value that will be used to select the fixed image during registration. This 
        should correspond to a value within the 'NormalizedSeriesDescription' column in
        the csv. If you have segmentation files in your data. They should correspond to
        this same series. Defaults to 'T1Post'.
        """
    ),
)

brain_preprocessing.add_argument(
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
    "--orientation",
    type=str,
    default="RAS",
    help=(
        "The orientation standard that you wish to set for preprocessed data. Defaults to 'RAS'."
    ),
)

brain_preprocessing.add_argument(
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
    "--no-skullstrip",
    action="store_true",
    help=(
        """
        Whether to not apply skullstripping to preprocessed data. Skullstripping will be
        applied if not specified."
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
    "-g",
    "--gpu",
    action="store_true",
    help=("Whether to use a gpu for Synthmorph registration. Defaults to False."),
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

debug_preprocessing = subparsers.add_parser(
    "debug-preprocessing",
    description=(
        """
        Preprocess NIfTI files for deep learning. A csv is required to
        indicate the location of source files and to procide the context
        for filenames. The outputs will comply with BIDS conventions.
        """
    ),
)

debug_preprocessing.add_argument(
    "preprocessed_dir",
    metavar="preprocessed-dir",
    type=Path,
    help=("The directory that will contain the preprocessed NIfTI files."),
)

debug_preprocessing.add_argument(
    "csv",
    type=Path,
    help=(
        """
        A csv containing nifti location and information required for the output file names.
        It must contain the columns: 'nifti', 'Anon_PatientID', 'Anon_StudyID', 
        'StudyInstanceUID', 'NormalizedSeriesDescription', and 'SeriesType'.
        """
    ),
)

debug_preprocessing.add_argument(
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


debug_preprocessing.add_argument(
    "--pipeline-key",
    type=str,
    default="debug",
    help=(
        """
        The key that will be used in the csv to indicate the new locations of preprocessed 
        files. Defaults to 'debug'.
        """
    ),
)

debug_preprocessing.add_argument(
    "--registration-key",
    type=str,
    default="T1Post",
    help=(
        """
        The value that will be used to select the fixed image during registration. This 
        should correspond to a value within the 'NormalizedSeriesDescription' column in
        the csv. If you have segmentation files in your data. They should correspond to
        this same series. Defaults to 'T1Post'.
        """
    ),
)

debug_preprocessing.add_argument(
    "--longitudinal-registration",
    action="store_true",
    help=(
        """
        Whether to use longitudinal registration. Additional studies for the same patient
        will be registered to the first study (chronologically). False if not specified.
        """
    ),
)

debug_preprocessing.add_argument(
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

debug_preprocessing.add_argument(
    "--orientation",
    type=str,
    default="RAS",
    help=(
        "The orientation standard that you wish to set for preprocessed data. Defaults to 'RAS'."
    ),
)

debug_preprocessing.add_argument(
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

debug_preprocessing.add_argument(
    "--no-skullstrip",
    action="store_true",
    help=(
        """
        Whether to not apply skullstripping to preprocessed data. Skullstripping will be
        applied if not specified."
        """
    ),
)

debug_preprocessing.add_argument(
    "-c",
    "--cpus",
    type=int,
    default=1,
    help=(
        "Number of cpus to use for multiprocessing. Defaults to 1 (no multiprocessing)."
    ),
)

debug_preprocessing.add_argument(
    "-g",
    "--gpu",
    action="store_true",
    help=("Whether to use a gpu for Synthmorph registration. Defaults to False."),
)


debug_preprocessing.add_argument(
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


def main():
    args = parser.parse_args()

    if args.command == "old-project-anon-keys":
        kwargs = {"input_dir": args.input_dir, "output_dir": args.output_dir}

        tracked_command(find_anon_keys, kwargs=kwargs, record_dir=args.output_dir)

    elif args.command == "reorganize-dicoms":
        kwargs = {
            "original_dicom_dir": args.original_dicom_dir,
            "new_dicom_dir": args.new_dicom_dir,
            "anon_csv": args.anon_csv,
            "cpus": args.cpus,
        }
        tracked_command(reorganize_dicoms, kwargs=kwargs, record_dir=args.new_dicom_dir)

    elif args.command == "dataset-to-nifti":
        kwargs = {
            "nifti_dir": args.nifti_dir,
            "csv": args.csv,
            "overwrite_nifti": args.overwrite,
        }
        tracked_command(
            convert_batch_to_nifti, kwargs=kwargs, record_dir=args.nifti_dir
        )

    elif args.command == "predict-series":
        if args.description_key is None:
            description_key = default_key
        else:
            with open(args.description_key, "r") as json_file:
                description_key = json.load(json_file)

        kwargs = {
            "csv": args.csv,
            "ruleset": args.ruleset,
            "description_key": description_key,
            "cpus": args.cpus,
        }
        tracked_command(series_from_csv, kwargs=kwargs, record_dir=args.csv.parent)

    elif args.command == "brain-preprocessing":
        kwargs = {
            "csv": args.csv,
            "preprocessed_dir": args.preprocessed_dir,
            "pipeline_key": args.pipeline_key,
            "registration_key": args.registration_key,
            "longitudinal_registration": args.longitudinal_registration,
            "registration_model": args.model,
            "orientation": args.orientation,
            "spacing": args.spacing,
            "skullstrip": not args.no_skullstrip,
            "cpus": args.cpus,
            "gpu": args.gpu,
            "verbose": args.verbose,
        }
        tracked_command(
            preprocess_from_csv, kwargs=kwargs, record_dir=args.preprocessed_dir
        )

    elif args.command == "debug-preprocessing":
        if isinstance(args.patients, str):
            args.patients = args.patients.split(",")

        kwargs = {
            "csv": args.csv,
            "preprocessed_dir": args.preprocessed_dir,
            "patients": args.patients,
            "pipeline_key": args.pipeline_key,
            "registration_key": args.registration_key,
            "longitudinal_registration": args.longitudinal_registration,
            "registration_model": args.registration_model,
            "orientation": args.orientation,
            "spacing": args.spacing,
            "skullstrip": not args.no_skullstrip,
            "cpus": args.cpus,
            "gpu": args.gpu,
            "verbose": args.verbose,
        }
        tracked_command(debug_from_csv, kwargs=kwargs, record_dir=args.preprocessed_dir)
