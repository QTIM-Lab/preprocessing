"""
The `__main__` module serves as the entrypoint for the `preprocessing` CLI. It has no public
functions intended for use in the Python API. Run 'preprocessing -h' in the terminal
for a command usage guide.
"""
import argparse
import sys
import os

from pathlib import Path

USAGE_STR = """
preprocessing <command> [<args>]

The following commands are available:
    validate-installation       Check that the `preprocessing` library is installed correctly along
                                with all of its dependencies.

    dicom-dataset               Create a DICOM dataset CSV compatible with subsequent `preprocessing`
                                scripts. The final CSV provides a series level summary of the location
                                of each series alongside metadata extracted from DICOM headers.  If the
                                previous organization schems of the dataset does not enforce a DICOM
                                series being isolated to a unique directory (instances belonging to
                                multiple series must not share the same lowest level directory),
                                reorganization must be applied for NIfTI conversion.

    nifti-dataset               Create a NIfTI dataset CSV compatible with subsequent `preprocessing`
                                scripts. The final CSV provides a series level summary of the location
                                of each series alongside metadata generated to simulate DICOM headers.
                                Specifically, ['PatientID', 'StudyDate', 'SeriesInstanceUID',
                                'SeriesDescription', 'StudyInstanceUID'] (and optionally
                                'NormalizedSeriesDescription') are inferred or randomly generated.

    dataset-to-nifti            Convert DICOMs to NIfTI file format. A CSV is required to map a
                                DICOM series to the resulting .nii.gz file and to provide
                                the context for filenames. The outputs will follow a BIDS inspired
                                convention.

    brain-preprocessing         Preprocess NIfTI files for deep learning. A CSV is required to
                                indicate the location of source files and to procide the context
                                for filenames. The outputs will follow a BIDS inspired convention.

    track-tumors                Longitudinal tracking of individual tumors. Each connected component
                                for a given label within a segmentation mask is assigned a unique ID
                                that will remain consistent across all scans belonging to the same
                                patient. This command assumes that longitudinal or atlas registration
                                was used when preprocessing the data.


Run `preprocessing <command> --help` for more details about how to use each individual command.

"""


parser = argparse.ArgumentParser(usage=USAGE_STR)

subparsers = parser.add_subparsers(dest="command")

validate_installation = subparsers.add_parser(
    "validate-installation",
    description=(
        """
        Check that the `preprocessing` library is installed correctly
        along with all of its dependencies.
        """
    )
)

dicom_dataset = subparsers.add_parser(
    "dicom-dataset",
    description=(
        """
        Create a DICOM dataset CSV compatible with subsequent `preprocessing`
        scripts. The final CSV provides a series level summary of the location
        of each series alongside metadata extracted from DICOM headers.  If the
        previous organization schems of the dataset does not enforce a DICOM
        series being isolated to a unique directory (instances belonging to
        multiple series must not share the same lowest level directory),
        reorganization must be applied for NIfTI conversion.
        """
    )
)

dicom_dataset.add_argument(
    "dicom_dir",
    metavar="dicom-dir",
    type=Path,
    help=("The directory in which the DICOM data is originally stored.")
)

dicom_dataset.add_argument(
    "csv",
    type=Path,
    help=(
        """
        The filepath of the output CSV which defines the constructed
        dataset. A corresponding instance level CSV will also be written
        out.
        """
    )
)

dicom_dataset.add_argument(
    "--reorg-dir",
    type=Path,
    default=None,
    help=(
        """
        The directory to which files are reorganized if a value
        other than `None` is provided. Defaults to `None`.
        """
    )
)

dicom_dataset.add_argument(
    "-a",
    "--anon",
    choices=["is_anon", "auto", "deferred"],
    default="auto",
    help=(
        """
        The anonymization scheme to apply to the completed CSV. Choose
        from:
            'is_anon'
                Assumes the data is already anonymized and uses the
                'PatientID' and 'StudyDate' values.

            'auto'
                Apply automated anonymization to the CSV. This function
                assumes that the 'PatientID' and 'StudyID' tags are
                consistent and correct to derive 'AnonPatientID' = 'sub_{i:02d}'
                and 'AnonStudyID' = 'ses_{i:02d}'.

            'deferred'
                Skip anonymization of the generated CSV. This step will be
                required for subsequent scripts.
        """
    )
)

dicom_dataset.add_argument(
    "-b",
    "--batch",
    type=int,
    default=1,
    help=("The size of the groups of files on which metadata extraction is applied.")
)

dicom_dataset.add_argument(
    "--assume-extension",
    action="store_true",
    help=("Assume that the DICOM instances all share the '.dcm' file extension.")
)

dicom_dataset.add_argument(
    "-m",
    "--mode",
    choices=["arbitrary", "midas"],
    default="arbitrary",
    help=(
        """
        The assumed data orgnaization scheme under `dicom_dir`. The choices
        are ['arbitrary', 'midas']. Defaults to 'arbitrary'.
        """
    ),
)

dicom_dataset.add_argument(
    "-c",
    "--cpus",
    type=int,
    default=1,
    help=(
        "Number of cpus to use for multiprocessing. Defaults to 1 (no multiprocessing)."
    ),
)

nifti_dataset = subparsers.add_parser(
    "nifti-dataset",
    description=(
        """
        Create a NIfTI dataset CSV compatible with subsequent `preprocessing`
        scripts. The final CSV provides a series level summary of the location
        of each series alongside metadata generated to simulate DICOM headers.
        Specifically, ['PatientID', 'StudyDate', 'SeriesInstanceUID',
        'SeriesDescription', 'StudyInstanceUID'] (and optionally
        'NormalizedSeriesDescription') are inferred or randomly generated.
        """
    )
)

nifti_dataset.add_argument(
    "nifti_dir",
    metavar="nifti-dir",
    type=Path,
    help=("The directory in which the DICOM data is originally stored.")
)

nifti_dataset.add_argument(
    "csv",
    type=Path,
    help=(
        """
        The filepath of the output CSV which defines the constructed
        dataset.
        """
    )
)

nifti_dataset.add_argument(
    "file_pattern",
    metavar="file-pattern",
    type=str,
    help=(
        """
        The file naming convention (without file extensions) of NIfTIs within a
        dataset. Variable names are encoded using '{}' (e.g. `file_pattern` =
        '{patient}_{study}_{series}' would find values for the `patient`, `study`,
        and `series` variables). The `patient`, `study`, and `series` variables
        must be defined.
        """
    )
)

nifti_dataset.add_argument(
    "-a",
    "--anon",
    choices=["is_anon", "auto", "deferred"],
    default="auto",
    help=(
        """
        The anonymization scheme to apply to the completed CSV. Choose
        from:
            'is_anon'
                Assumes the data is already anonymized and uses the
                'PatientID' and 'StudyDate' values.

            'auto'
                Apply automated anonymization to the CSV. This function
                assumes that the 'PatientID' and 'StudyID' tags are
                consistent and correct to derive 'AnonPatientID' = 'sub_{i:02d}'
                and 'AnonStudyID' = 'ses_{i:02d}'.

            'deferred'
                Skip anonymization of the generated CSV. This step will be
                required for subsequent scripts.
        """
    )
)

nifti_dataset.add_argument(
    "-b",
    "--batch",
    type=int,
    default=20,
    help=("The size of the groups of files on which metadata extraction is applied.")
)

nifti_dataset.add_argument(
    "-ss",
    "--seg-series",
    type=str,
    default=None,
    help=(
        """
        The series description of segmentations within the dataset, assuming a
        consistent value is present. Must also specify `seg_target` to handle
        segmentations properly. Defaults to `None`.
        """
    )
)

nifti_dataset.add_argument(
    "-st",
    "--seg-target",
    type=str,
    default=None,
    help=(
        """
        The series description of the series from which segmentations are derived,
        assuming a consistent value is present. Must also specify `seg_series` to
        handle segmentations properly. Defaults to `None`.
        """
    )
)

nifti_dataset.add_argument(
    "-c",
    "--cpus",
    type=int,
    default=1,
    help=(
        "Number of cpus to use for multiprocessing. Defaults to 1 (no multiprocessing)."
    ),
)

dataset_to_nifti = subparsers.add_parser(
    "dataset-to-nifti",
    description=(
        """
        Convert DICOMs to NIfTI file format. A CSV is required to map a
        DICOM series to the resulting .nii.gz file and to provide
        the context for filenames. The outputs will follow a BIDS
        inspired convention.
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
        names. It must contain the columns: ['Dicoms', 'AnonPatientID',
        'AnonStudyID', 'StudyInstanceUID', 'SeriesInstanceUID', 'Manufacturer', 'NormalizedSeriesDescription',
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
    "--skip-integrity-checks",
    action="store_true",
    help="Whether to skip performing the DICOM integrity checks before attempting conversion to NIfTI.",
)

dataset_to_nifti.add_argument(
    "-t",
    "--conversion-tolerance",
    type=float,
    help="The conversion tolerance for `highdicom`'s NIfTI conversion. Defaults to 0.05.",
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
    default="Preprocessed",
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
    choices=["rigid", "affine", "affine_crop", "joint", "deform"],
    default="affine",
    help=(
        """
        The synthmorph model that will be used to perform registration. Choices are: 'rigid', 'affine', 'affine_crop', 'joint',
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

tumor_tracking = subparsers.add_parser(
    "track-tumors",
    description=(
        """
        Longitudinal tracking of individual tumors. Each connected component for a given
        label within a segmentation mask is assigned a unique ID that will remain consistent
        across all scans belonging to the same patient. This command assumes that
        longitudinal or atlas registration was used when preprocessing the data.
        """
    ),
)

tumor_tracking.add_argument(
    "tracking_dir",
    metavar="tracking-dir",
    type=Path,
    help=("The directory that will contain the tumor id mask files."),
)

tumor_tracking.add_argument(
    "csv",
    type=Path,
    help=(
        """
        A CSV containing NIfTI location and information required for the output file names.
        It must contain the columns: 'AnonPatientID', 'AnonStudyID', and 'SeriesType'.
        Additionally, '<pipeline_key>Seg' must be present with the assumption that the
        corresponding segmentation masks have been preprocessed.
        """
    ),
)

tumor_tracking.add_argument(
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

tumor_tracking.add_argument(
    "-pk",
    "--pipeline-key",
    type=str,
    default="preprocessed",
    help=(
        """
        The key used in the CSV when preprocessing was performed. Defaults to 'preprocessed'.
        """
    ),
)

tumor_tracking.add_argument(
    "-l",
    "--labels",
    type=str,
    default="1",
    help=(
        """
        A comma delimited list of the labels included in the segmentation masks.
        """
    ),
)

tumor_tracking.add_argument(
    "-c",
    "--cpus",
    type=int,
    default=1,
    help=(
        "Number of cpus to use for multiprocessing. Defaults to 1 (no multiprocessing)."
    ),
)

volume_tracking = subparsers.add_parser(
    "track-volume",
    description=(
        """
        Longitudinal tracking of individual tumor volumes. Each connected component for a given
        label within a segmentation mask is assigned a unique ID that will remain consistent
        across all scans belonging to the same patient. This command assumes that
        longitudinal or atlas registration was used when preprocessing the data.
        """
    ),
)

volume_tracking.add_argument(
    "plot_dir",
    metavar="plot-dir",
    type=Path,
    help=("The directory that will contain the tumor id mask files."),
)

volume_tracking.add_argument(
    "csv",
    type=Path,
    help=(
        """
        A CSV containing NIfTI location and information required for the output file names.
        It must contain the columns: 'AnonPatientID', 'AnonStudyID', and 'SeriesType'.
        Additionally, '<pipeline_key>Seg' must be present with the assumption that the
        corresponding segmentation masks have been preprocessed.
        """
    ),
)

volume_tracking.add_argument(
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

volume_tracking.add_argument(
    "-pk",
    "--pipeline-key",
    type=str,
    default="preprocessed",
    help=(
        """
        The key used in the CSV when preprocessing was performed. Defaults to 'preprocessed'.
        """
    ),
)

volume_tracking.add_argument(
    "-c",
    "--cpus",
    type=int,
    default=1,
    help=(
        "Number of cpus to use for multiprocessing. Defaults to 1 (no multiprocessing)."
    ),
)


def main() -> None:
    """
    The CLI for the `preprocessing` library. Run 'preprocessing -h' for additional help.
    """
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_usage()
        exit(0)

    assert not any(["SLURM" in var for var in os.environ.keys()]), (
        "This is an incorrect use of the `preprocessing` library's CLI. To use with slurm, "
        "please switch to the `spreprocessing` CLI. Run `spreprocessing -h` for additional help."
    )

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
        char_limit=None
    )
    def tracked_command(
        func: Callable, kwargs: Dict[str, Any], record_dir: Path | str
    ) -> Any:
        """
        A standardized command format that runs a function from the `preprocessing` library
        and tracks useful information with pycrumbs.

        Parameters
        ----------
        func: Callable
            The function that will be used for the command.

        kwargs: Dict[str, Any]
            The key-word arguments that will be passed to `func`.

        record_dir: Path | str
            The directory in which the pycrumbs tracking json will be stored (should be the
            same as the directory containing `func`'s outputs').

        Returns
        -------
        output: Any
            Whatever is returned by `func(**kwargs)`.
        """
        return func(**kwargs)

    if args.command == "validate-installation":
        try:
            # check all packages
            from preprocessing import data
            from preprocessing import brain
            from preprocessing import qc
            from preprocessing import constants
            from preprocessing import dcm_tools
            from preprocessing import synthmorph
            from preprocessing import utils

            if "PREPROCESSING_MODELS_PATH" in os.environ:
                pass

            else:
                raise Exception("$PREPROCESSING_MODELS_PATH not specified.")

            utils.check_for_models(os.environ["PREPROCESSING_MODELS_PATH"])
            print("`preprocessing` installation is valid.")

        except Exception as error:
            raise Exception(f"`preprocessing` installation is invalid. Encountered the following exception during validation: {error}")

    elif args.command == "dicom-dataset":
        from preprocessing.data import create_dicom_dataset

        kwargs = {
            "dicom_dir": args.dicom_dir,
            "dataset_csv": args.csv,
            "reorg_dir": args.reorg_dir,
            "anon": args.anon,
            "batch_size": args.batch,
            "file_extension": "*.dcm" if args.assume_extension else "*",
            "mode": args.mode,
            "cpus": args.cpus
        }

        tracked_command(
            create_dicom_dataset, kwargs=kwargs, record_dir=args.csv.parent
        )

    elif args.command == "nifti-dataset":
        from preprocessing.data import create_nifti_dataset

        kwargs = {
            "nifti_dir": args.nifti_dir,
            "dataset_csv": args.csv,
            "anon": args.anon,
            "batch_size": args.batch,
            "processor_kwargs": {
                "file_pattern": args.file_pattern,
                "seg_series": args.seg_series,
                "seg_target": args.seg_target
            },
            "cpus": args.cpus
        }

        tracked_command(
            create_nifti_dataset, kwargs=kwargs, record_dir=args.csv.parent
        )

    elif args.command == "dataset-to-nifti":
        from preprocessing.data import convert_batch_to_nifti

        kwargs = {
            "nifti_dir": args.nifti_dir,
            "csv": args.csv,
            "overwrite_nifti": args.overwrite,
            "skip_integrity_checks": args.skip_integrity_checks,
            "tolerance": args.conversion_tolerance,
            "cpus": args.cpus
        }

        tracked_command(
            convert_batch_to_nifti, kwargs=kwargs, record_dir=args.nifti_dir
        )

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
            "atlas_target": args.atlas_target,
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

        tracked_command(preprocess_from_csv, kwargs=kwargs, record_dir=args.preprocessed_dir)

    elif args.command == "track-tumors":
        from preprocessing.qc import track_tumors_csv

        if isinstance(args.patients, str):
            args.patients = args.patients.split(",")

        if isinstance(args.labels, str):
            args.labels = [int(i) for i in args.labels.split(",")]

        kwargs = {
            "csv": args.csv,
            "tracking_dir": args.tracking_dir,
            "patients": args.patients,
            "pipeline_key": args.pipeline_key,
            "labels": args.labels,
            "cpus": args.cpus,
        }

        tracked_command(track_tumors_csv, kwargs=kwargs, record_dir=args.tracking_dir)

    elif args.command == "track-volume":
        from preprocessing.qc import vol_plot_csv

        if isinstance(args.patients, str):
            args.patients = args.patients.split(",")

        kwargs = {
            "csv": args.csv,
            "plot_dir": args.plot_dir,
            "patients": args.patients,
            "pipeline_key": args.pipeline_key,
            "cpus": args.cpus,
        }

        tracked_command(vol_plot_csv, kwargs=kwargs, record_dir=args.plot_dir)

    exit(0)


__all__ = []
