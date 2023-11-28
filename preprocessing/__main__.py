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
            help=(
                "A csv mapping PatientID and StudyInstanceUID to " "anonymous values."
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

        reorganize_dicoms(
            original_dicom_dir=args.original_dicom_dir,
            new_dicom_dir=args.new_dicom_dir,
            anon_csv=args.anon_csv,
            cpus=args.cpus,
        )


def main():
    preprocessing_cli()
