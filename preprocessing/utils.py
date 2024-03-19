"""
The `utils` module contains custom exceptions and useful functions that are referenced
frequently throughout the rest of the library.

Public Classes
______________
MissingColumnsError
    Exception to be raised in the cases where a DataFrame doesn't have the necessary
    columns to run a function.

MissingSoftwareError
    Exception to be raised in the cases where software external to Python is called and
    that software has not been properly sourced.

Public Functions
________________
check_required_columns
    Checks DataFrames for the presence of required columns.

source_external_software
    Sources external software from the paths where they are located on Martinos Machines.
    If the paths do not exist, the required software is checked using 'shutil.which'.

sitk_to_surfa
    Convert a SimpleITK.Image to a surfa.Volume.

surfa_to_sitk
    Convert a surfa.Volume to a SimpleITK.Image.

initialize_models
    Set the path to the locations where the preprocessing models are (or will be stored).

check_for_models
    Checks that all of the Synthmorph and Synthstrip models have been successfully installed.
"""

import os
import pandas as pd
import numpy as np

from shutil import which
from typing import Sequence
from SimpleITK import (
    Image,
    GetImageFromArray,
    GetArrayFromImage,
)
from surfa import Volume, ImageGeometry
from subprocess import run


class MissingSoftwareError(Exception):
    """
    Exception to be raised in the cases where software external to Python is called and that software has
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

    if os.path.exists(
        "/usr/pubsw/packages/fsl/6.0.6/bin"
    ):  # Source on Martinos Machine
        os.environ["PATH"] = ("/usr/pubsw/packages/fsl/6.0.6/bin:") + os.environ.get(
            "PATH", "/usr/bin/"
        )

    else:
        required_software = ["dcm2niix"]
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
                f"The required column '{missing_column}' is not present in the provided CSV/DataFrame. "
                f"To run this command, you must include the following columns: {required_columns}."
            )
        else:
            super().__init__(
                f"The required column '{missing_column}' is not present in the provided CSV/DataFrame. "
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
        A DataFrame that must contain all od the columns in 'required_columns'.

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


# assumes 3D images
def sitk_to_surfa(sitk_im: Image) -> Volume:
    """
    Convert a SimpleITK.Image to a surfa.Volume.

    Parameters
    __________
    sitk_im: Image
        A SimpleITK.Image to convert.

    Returns
    _______
    sf_im: Volume
        The corresponding surfa.Volume.
    """
    lps_ras = np.diag([-1, -1, 1])
    data = GetArrayFromImage(sitk_im).transpose(2, 1, 0)
    spacing = sitk_im.GetSpacing()

    vox2world = np.eye(4)
    vox2world[:3, :3] = (
        lps_ras
        @ np.reshape(sitk_im.GetDirection(), (3, 3))
        * np.reshape(spacing * 3, (3, 3))
    )
    vox2world[:3, 3] = lps_ras @ np.array(sitk_im.GetOrigin())

    geom = ImageGeometry(shape=data.shape, voxsize=spacing, vox2world=vox2world)

    return Volume(data, geom)


def surfa_to_sitk(sf_im: Volume) -> Image:
    """
    Convert a surfa.Volume to a SimpleITK.Image.

    Parameters
    __________
    sf_im: Volume
        A surfa.Volume to convert.

    Returns
    _______
    sitk_im: Image
        The corresponding SimpleITK.Image.
    """

    ras_lps = np.diag([-1, -1, 1])
    data = np.array(sf_im).transpose(2, 1, 0)

    if data.dtype == np.bool_:
        data = data.astype(int)

    spacing = sf_im.geom.voxsize.tolist()

    sitk_im = GetImageFromArray(data)

    sitk_im.SetSpacing(spacing)

    sitk_im.SetDirection(
        (
            ras_lps @ sf_im.geom.vox2world[:3, :3] / np.reshape(spacing * 3, (3, 3))
        ).flatten()
    )

    sitk_im.SetOrigin(ras_lps @ sf_im.geom.vox2world[:3, 3])

    return sitk_im


def initialize_models() -> str:
    """
    Set the path to the locations where the preprocessing models are (or will be stored).

    Returns
    _______
    models_dir
        The directory specified by user input to store the models.
    """
    models_dir = input(
        """
The `preprocessing` library utilizes external models from Synthstrip and Synthmorph.
The directory that contains these files is specified using the 'PREPROCESSING_MODELS_PATH',
which is not currently defined. If you are on a Martinos Machine, please enter 'Martinos' below.
Otherwise, enter the path where you wish to store the Synthstrip and Synthmorph models
(this directory will be created if it does not yet exist):\n
"""
    )

    if models_dir.lower() == "martinos":
        models_dir = "/autofs/vast/qtim/tools/preprocessing_models"

    add_to_rc = input(
        f"\nWould you like to add 'export PREPROCESSING_MODELS_PATH={models_dir}' to your RC file? (y/n):\n\n"
    )

    if add_to_rc.lower() == "y":
        supported_rc_files = [
            os.environ["HOME"] + f"/{name}" for name in [".bashrc", ".zshrc"]
        ]

        rc_updated = False

        for rc_file in supported_rc_files:
            if os.path.exists(rc_file):
                contents = open(rc_file, "r").read()
                contents += f"\nexport PREPROCESSING_MODELS_PATH={models_dir}"
                open(rc_file, "w").write(contents)

                rc_updated = True
                break

        if not rc_updated:
            print(
                f"Your RC file could not be found among {supported_rc_files}. You will have to update it manually"
            )

        os.environ["PREPROCESSING_MODELS_PATH"] = models_dir
        check_for_models(models_dir)

    return models_dir


def check_for_models(models_dir: str) -> None:
    """
    Checks that all of the Synthmorph and Synthstrip models have been successfully installed.

    Parameters
    __________
    models_dir: str
        The directory that should (or will) contain the preprocessing models.

    Returns
    _______
    None
        If the models are already installed, nothing will happen. Otherwise, they will be installed.
        If the installation process encounters an error, this function prints the error to the console
        and quits Python.

    """
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(f"{models_dir}/synthmorph.affine.2.h5"):
        command = f"wget -O {models_dir}/synthmorph.affine.2.h5 https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/repo/annex.git/annex/objects/66d/dba/SHA256E-s51455312--1ac5304b683036e5177f5b4ad38fa09fcbbe7883e742d6fa5bdaedd0e619ced6.2.h5/SHA256E-s51455312--1ac5304b683036e5177f5b4ad38fa09fcbbe7883e742d6fa5bdaedd0e619ced6.2.h5"
        print(command)
        result = run(command.split(" "))

        try:
            result.check_returncode()

        except Exception as error:
            print(f"synthmorph.affine.2.h5 could not be downloaded due to {error}")
            quit()

    if not os.path.exists(f"{models_dir}/synthmorph.deform.2.h5"):
        command = f"wget -O {models_dir}/synthmorph.deform.2.h5 https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/repo/annex.git/annex/objects/b6d/a4d/SHA256E-s3508630384--0cfb8f7a5073f15efba6e2108c3c6e05cc2ef28d21a94822150df7a760afe156.2.h5/SHA256E-s3508630384--0cfb8f7a5073f15efba6e2108c3c6e05cc2ef28d21a94822150df7a760afe156.2.h5"
        print(command)
        result = run(command.split(" "))

        try:
            result.check_returncode()

        except Exception as error:
            print(f"synthmorph.deform.2.h5 could not be downloaded due to {error}")
            quit()

    if not os.path.exists(f"{models_dir}/synthmorph.rigid.1.h5"):
        command = f"wget -O {models_dir}/synthmorph.rigid.1.h5 https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/repo/annex.git/annex/objects/b6d/a4d/SHA256E-s3508630384--0cfb8f7a5073f15efba6e2108c3c6e05cc2ef28d21a94822150df7a760afe156.2.h5/SHA256E-s3508630384--0cfb8f7a5073f15efba6e2108c3c6e05cc2ef28d21a94822150df7a760afe156.2.h5"
        print(command)
        result = run(command.split(" "))

        try:
            result.check_returncode()

        except Exception as error:
            print(f"synthmorph.rigid.1.h5 could not be downloaded due to {error}")
            quit()

    if not os.path.exists(f"{models_dir}/synthstrip.1.pt"):
        command = f"wget -O {models_dir}/synthstrip.1.pt https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/repo/annex.git/annex/objects/fdd/51c/SHA256E-s30851709--37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33.1.pt/SHA256E-s30851709--37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33.1.pt"
        print(command)
        result = run(command.split(" "))

        try:
            result.check_returncode()

        except Exception as error:
            print(f"synthstrip.1.pt could not be downloaded due to {error}")
            quit()

    if not os.path.exists(f"{models_dir}/synthstrip.nocsf.1.pt"):
        command = f"wget -O {models_dir}/synthstrip.nocsf.1.pt https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/repo/annex.git/annex/objects/ae1/f00/SHA256E-s30851709--62bf01137c45b5f0cc04d59dbaed5b9ac138b3f25b766c062a7c1a0d696ecb28.1.pt/SHA256E-s30851709--62bf01137c45b5f0cc04d59dbaed5b9ac138b3f25b766c062a7c1a0d696ecb28.1.pt"
        print(command)
        result = run(command.split(" "))

        try:
            result.check_returncode()

        except Exception as error:
            print(f"synthstrip.nocsf.1.pt could not be downloaded due to {error}")
            quit()


__all__ = [
    "MissingColumnsError",
    "MissingSoftwareError",
    "check_required_columns",
    "source_external_software",
    "sitk_to_surfa",
    "surfa_to_sitk",
    "initialize_models",
    "check_for_models",
]
