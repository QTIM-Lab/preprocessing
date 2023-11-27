import argparse
import sys


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

        find_anon_keys(args.input_dir, args.output_dir)


def main():
    preprocessing_cli()
