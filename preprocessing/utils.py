import os
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from shutil import which
from typing import Sequence


class MissingSoftwareError(Exception):
    """
    Exception to be raised in the cases where software external to python is called and that software has
    not been properly sourced.
    """

    def __init__(self, software: str, required_software: Sequence[str]):
        """
        Parameters
        __________
        software: str
            The name of the software that is required but cannot be found.
        required_software: Sequence[str]
            A sequence of the names of all of the required software that must be installed.

        Returns
        _______
        None:
            If raised, a message indicating the software that cannot be found and the list of all required
            software will be printed.
        """
        super().__init__(
            f"The required software '{software}' is not installed. This library depends on: {required_software}. "
            "Please ensure these are installed and sourced correctly if you are not on a Martinos machine."
        )


def source_external_software():
    """
    Sources external software from the paths where they are located on Martinos Machines. If the paths do not exist,
    the required software is checked using 'shutil.which'.

    Parameters
    __________
    None

    Returns
    _______
    None
        'MissingSoftwareError' is raised if any software is not available.
    """
    paths = [
        "/usr/local/freesurfer/dev/bin",
        "/usr/pubsw/packages/fsl/6.0.6/bin",
        "/usr/pubsw/packages/slicer/Slicer-5.2.2-linux-amd64/",
        "/usr/pubsw/packages/ANTS/2.3.5/bin",
        "/usr/pubsw/packages/CUDA/11.8/bin",
    ]
    if all(os.path.exists(path) for path in paths):  # Source on Martinos Machine
        os.environ["PATH"] = (
            "/usr/local/freesurfer/dev/bin:"
            "/usr/pubsw/packages/fsl/6.0.6/bin:"
            "/usr/pubsw/packages/slicer/Slicer-5.2.2-linux-amd64/:"
            "/usr/pubsw/packages/ANTS/2.3.5/bin:"
            "/usr/pubsw/packages/CUDA/11.8/bin:"
        ) + os.environ["PATH"]

        os.environ["LD_LIBRARY_PATH"] = (
            "/usr/pubsw/packages/CUDA/11.8/lib64:" + os.environ["LD_LIBRARY_PATH"]
        )

        os.environ["ANTSPATH"] = "/usr/pubsw/packages/ANTS/2.3.5/bin"

        os.environ["FSLDIR"] = "/usr/pubsw/packages/fsl/6.0.6/"
        os.system(f"source {os.environ['FSLDIR']}/etc/fslconf/fsl.sh")

        os.environ["FREESURFER_HOME"] = "/usr/local/freesurfer/dev"
        os.system(f"source {os.environ['FREESURFER_HOME']}/SetUpFreeSurfer.sh")

    else:
        required_software = ["dcm2niix", "Slicer", "ANTS"]
        for software in required_software:
            if which(software) is None:
                raise MissingSoftwareError(software, required_software)


class MissingColumnsError(Exception):
    """
    Exception to be raised in the cases where a DataFrame doesn't have the necessary columns to run a function.
    """

    def __init__(
        self,
        missing_column: str,
        required_columns: Sequence[str],
        optional_columns: Sequence[str] | None = None,
    ):
        """
        Parameters
        __________
        missing_column: str
            The name of a required column that is not in the DataFrame.
        required_software: Sequence[str]
            A sequence of the names of all of the required columns for a given function.
        required_software: Sequence[str]
            A sequence of the names of columns that are not required but provide additional behavior if
            provided.

        Returns
        _______
        None:
            If raised, a message indicating the column that cannot be found, the list of all required
            columns, and any optional columns will be printed.
        """
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
    """
    Checks DataFrames for the presence of required columns.

    Parameters
    __________
    df: pd.DataFrame
        A DataFrame that must contain all od the columns in 'required_columns'
    required_columns: Sequence[str]
        A sequence of the names of columns that will be required to run subsequent code.
    optional_columns: Sequence[str] | None
        A sequence of the names of columns that have additional behavior if provided, but are not required.
        Defaults to None.

    Returns
    _______
    None
        'MissingColumnsError' is raised if any required columns are not in the DataFrame.
    """

    for column in required_columns:
        if column not in df.keys():
            raise MissingColumnsError(
                missing_column=column,
                required_columns=required_columns,
                optional_columns=optional_columns,
            )


def sitk_check(file: Path | str):
    sitk_image = sitk.ReadImage(file)
    sitk.WriteImage(sitk_image, file)
