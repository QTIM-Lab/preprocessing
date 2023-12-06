import argparse
import sys

from pathlib import Path


class preprocessing_cli(object):
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Useful commands for QTIM data preprocessing",
            usage="""preprocessing <command> [<args>]

The following commands are available:
    old_project_anon_keys       Create anonymization keys for anonymous PatientID and VisitID
                                from previous QTIM organizational scheme. Should be compatible
                                with data following a following <Patient_ID>/<Study_ID> directory
                                hierarchy.

    reorganize_dicoms           Reorganize DICOMs to follow the BIDS convention. Any DICOMs found
                                recursively within this directory will be reorganized (at least 
                                one level of subdirectories is assumed). Anonomyzation keys for
                                PatientIDs and StudyIDs are provided within a csv.

    dataset_to_nifti            Convert DICOMs to NIfTI file format. A csv is required to map a
                                DICOM series to the resulting .nii.gz file and to provide
                                the context for filenames. The outputs will comply with the BIDS
                                conventions.

    brain_preprocessing         Preprocess NIfTI files for deep learning. A csv is required to 
                                indicate the location of source files and to procide the context 
                                for filenames. The outputs will comply with BIDS conventions.

Run `preprocessing <command> --help` for more details about how to use each individual command. 

""",
        )

        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])

        if not hasattr(self, args.command):
            print("That command is not supported.")
            parser.print_help()
            exit(1)

        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def old_project_anon_keys(self):
        from preprocessing.bids import find_anon_keys

        parser = argparse.ArgumentParser(
            description=(
                "Create anonymization keys for anonymous PatientID and StudyID "
                "from the previous QTIM organizational scheme. Is compatible "
                "with data following a following <Patient_ID>/<Study_ID> directory "
                "hierarchy."
            )
        )

        parser.add_argument(
            "input_dir",
            type=str,
            help=(
                "The directory containing all of the dicom files for a project. "
                "Should follow the <Patient_ID>/<Study_ID> convention"
            ),
        )

        parser.add_argument(
            "output_dir",
            type=str,
            help=(
                "The directory that will contain the output csv and "
                "potentially an error file."
            ),
        )

        args = parser.parse_args(sys.argv[2:])

        find_anon_keys(input_dir=args.input_dir, output_dir=args.output_dir)

    def reorganize_dicoms(self):
        from preprocessing.bids import reorganize_dicoms

        parser = argparse.ArgumentParser()

        parser.add_argument(
            "original_dicom_dir",
            type=Path,
            help=(
                "The directory containing all of the DICOM files you wish to "
                "organize."
            ),
        )

        parser.add_argument(
            "new_dicom_dir",
            type=Path,
            help=(
                "The directory that will contain the same DICOM files "
                "reorganized to follow the BIDS convention."
            ),
        )

        parser.add_argument(
            "anon_csv",
            type=Path,
            help=("A csv mapping PatientID and StudyInstanceUID to anonymous values."),
        )

        parser.add_argument(
            "-c",
            "--cpus",
            type=int,
            default=0,
            help=(
                "Number of cpus to use for multiprocessing. Defaults "
                "to 0 (no multiprocessing)."
            ),
        )

        args = parser.parse_args(sys.argv[2:])

        reorganize_dicoms(
            original_dicom_dir=args.original_dicom_dir,
            new_dicom_dir=args.new_dicom_dir,
            anon_csv=args.anon_csv,
            cpus=args.cpus,
        )

    def validate_bids(self):
        from preprocessing.bids import validate

        paths = sys.argv[2:]

        if ("--help" in paths) or ("-h" in paths):
            print(
                "File(s) validated for BIDS convention. Use spaces to delimit multiple files."
            )
        else:
            validate(paths)

    def dataset_to_nifti(self):
        from preprocessing.bids import convert_batch_to_nifti

        parser = argparse.ArgumentParser()

        parser.add_argument(
            "nifti_dir",
            type=Path,
            help=("The directory that will contain the converted NIfTI files."),
        )

        parser.add_argument(
            "csv",
            type=Path,
            help=(
                "A csv containing dicom location and information required "
                "for the nifti file names. It must contain the columns: "
                "'dicoms', 'Anon_PatientID', 'Anon_StudyID', 'StudyInstanceUID', "
                "'NormalizedSeriesDescription', and 'SeriesType'."
            ),
        )

        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Whether to overwrite the .nii.gz files. False if not specified.",
        )

        parser.add_argument(
            "-c",
            "--cpus",
            type=int,
            default=0,
            help=(
                "Number of cpus to use for multiprocessing. Defaults "
                "to 0 (no multiprocessing)."
            ),
        )

        args = parser.parse_args(sys.argv[2:])

        convert_batch_to_nifti(
            nifti_dir=args.nifti_dir, csv=args.csv, overwrite_nifti=args.overwrite
        )

    def brain_preprocessing(self):
        from preprocessing.brain import preprocess_from_csv

        parser = argparse.ArgumentParser()

        parser.add_argument(
            "preprocessed_dir",
            type=Path,
            help=("The directory that will contain the preprocessed NIfTI files."),
        )

        parser.add_argument(
            "csv",
            type=Path,
            help=(
                "A csv containing nifti location and information required "
                "for the output file names. It must contain the columns: "
                "'nifti', 'Anon_PatientID', 'Anon_StudyID', 'StudyInstanceUID', "
                "'NormalizedSeriesDescription', and 'SeriesType'."
            ),
        )

        parser.add_argument(
            "--pipeline_key",
            type=str,
            default="preprocessed",
            help=(
                "The key that will be used in the csv to indicate the new locations "
                "of preprocessed files. Defaults to 'preprocessed'."
            ),
        )

        parser.add_argument(
            "--longitudinal_registration",
            action="store_true",
            help=(
                "Whether to use longitudinal registration. Additional studies "
                "for the same patient will be registered to the first study "
                "(chronologically). False if not specified."
            ),
        )

        parser.add_argument(
            "--orientation",
            type=str,
            default="RAI",
            help=(
                "The orientation standard that you wish to set for preprocessed "
                "data. Defaults to 'RAI'."
            ),
        )

        parser.add_argument(
            "--spacing",
            type=str,
            default="1,1,1",
            help=(
                "A comma delimited list indicating the desired spacing of preprocessed "
                "data. Measurements are in mm. Defaults to '1,1,1'."
            ),
        )

        parser.add_argument(
            "--no_skullstrip",
            action="store_true",
            help=(
                "Whether to not apply skullstripping to preprocessed data. "
                "Skullstripping will be applied if not specified."
            ),
        )

        parser.add_argument(
            "-c",
            "--cpus",
            type=int,
            default=0,
            help=(
                "Number of cpus to use for multiprocessing. Defaults "
                "to 0 (no multiprocessing)."
            ),
        )

        args = parser.parse_args(sys.argv[2:])

        preprocess_from_csv(
            csv=args.csv,
            preprocessed_dir=args.preprocessed_dir,
            pipeline_key=args.pipeline_key,
            longitudinal_registration=args.longitudinal_registration,
            orientation=args.orientation,
            spacing=args.spacing,
            skullstrip=not args.no_skullstrip,
            cpus=args.cpus,
        )


def main():
    preprocessing_cli()
