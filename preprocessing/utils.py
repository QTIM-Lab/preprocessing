import os
import pandas as pd
from shutil import which
from typing import Sequence


class MissingSoftwareError(Exception):
    def __init__(self, software: str, required_software: Sequence[str]):
        super().__init__(
            f"The required software '{software}' is not installed. This library depends on: {required_software}. "
            "Please ensure these are installed and sourced correctly if you are not on a Martinos machine."
        )


def source_external_software():
    paths = [
        # "/usr/local/freesurfer/7.3.3/bin",
        "/usr/pubsw/packages/fsl/6.0.6/bin",
        "/usr/pubsw/packages/slicer/Slicer-5.2.2-linux-amd64/",
        "/usr/pubsw/packages/ANTS/2.3.5/bin",
    ]
    if all(os.path.exists(path) for path in paths):  # Source on Martinos Machine
        os.environ["PATH"] += (
            # ":/usr/local/freesurfer/7.3.3/bin"
            ":/usr/pubsw/packages/fsl/6.0.6/bin"
            ":/usr/pubsw/packages/slicer/Slicer-5.2.2-linux-amd64/"
            ":/usr/pubsw/packages/ANTS/2.3.5/bin"
        )
        os.environ["ANTSPATH"] = "/usr/pubsw/packages/ANTS/2.3.5/bin"

        # os.environ["FREESURFER_HOME"] = "/usr/local/freesurfer/7.3.3"
        # os.system(f"source {os.environ['FREESURFER_HOME']}/SetUpFreeSurfer.sh")

        os.environ["FSLDIR"] = "/usr/pubsw/packages/fsl/6.0.6/"
        os.system(f"source {os.environ['FSLDIR']}/etc/fslconf/fsl.sh")

    else:
        required_software = ["dcm2niix", "Slicer", "ANTS"]
        for software in required_software:
            if which(software) is None:
                raise MissingSoftwareError(software, required_software)


class MissingColumnsError(Exception):
    def __init__(
        self,
        missing_column: str,
        required_columns: Sequence[str],
        optional_columns: Sequence[str] | None = None,
    ):
        if optional_columns is None:
            super().__init__(
                f"The required column '{missing_column}' is not present in the provided csv. "
                f"To run this command, you must include the following columns: {required_columns}."
            )
        else:
            super().__init__(
                f"The required column '{missing_column}' is not present in the provided csv. "
                f"To run this command, you must include the following columns: {required_columns}. "
                f"You may optionally wish to include the following columns as well: {optional_columns}"
            )


def check_required_columns(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    optional_columns: Sequence[str] | None = None,
):
    for column in required_columns:
        if column not in df.keys():
            raise MissingColumnsError(
                missing_column=column,
                required_columns=required_columns,
                optional_columns=optional_columns,
            )
