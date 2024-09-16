"""
The `utils` module contains custom exceptions and useful functions that are referenced
frequently throughout the rest of the library.

Public Classes
--------------
MissingColumnsError
    Exception to be raised in the cases where a DataFrame doesn't have the necessary
    columns to run a function.

MissingSoftwareError
    Exception to be raised in the cases where software external to Python is called and
    that software has not been properly sourced.

Public Functions
----------------
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

cpu_adjust
    Confirms that the number of parallel processes will fit into the current amount of allowed
    memory and adjusts to an appropriate number if necessary.

update_errorfile
    Updates an errorfile with the full traceback of an encountered exception and
    records the corresponding function and arguments.

parse_string
    Extracts variables from a string following a consistent pattern indicated using
    '{}'. Returns a dictionary with keys corresponding to the variable names.
"""

import os
import pandas as pd
import numpy as np
import psutil
import warnings
import traceback
import re

from shutil import which
from typing import Sequence, Dict, Any
from SimpleITK import (
    Image,
    GetImageFromArray,
    GetArrayFromImage,
)
from surfa import Volume, ImageGeometry
from subprocess import run
from pathlib import Path
from multiprocessing import Process, Queue
from itertools import islice
from time import sleep


class MissingSoftwareError(Exception):
    """
    Exception to be raised in the cases where software external to Python is called and that software has
    not been properly sourced.
    """

    def __init__(self, software: str, required_software: Sequence[str]):
        """
        Parameters
        ----------
        software: str
            The name of the software that is required but cannot be found.
        required_software: Sequence[str]
            A sequence of the names of all of the required software that must be installed.

        Returns
        -------
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
    ----------
    None

    Returns
    -------
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
        ----------
        missing_column: str
            The name of a required column that is not in the DataFrame.

        required_software: Sequence[str]
            A sequence of the names of all of the required columns for a given function.

        required_software: Sequence[str]
            A sequence of the names of columns that are not required but provide additional behavior if
            provided.

        Returns
        -------
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
    ----------
    df: pd.DataFrame
        A DataFrame that must contain all od the columns in 'required_columns'.

    required_columns: Sequence[str]
        A sequence of the names of columns that will be required to run subsequent code.

    optional_columns: Sequence[str] | None
        A sequence of the names of columns that have additional behavior if provided, but are not required.
        Defaults to None.

    Returns
    -------
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
    ----------
    sitk_im: Image
        A SimpleITK.Image to convert.

    Returns
    -------
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
    ----------
    sf_im: Volume
        A surfa.Volume to convert.

    Returns
    -------
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
    -------
    models_dir: str
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
    ----------
    models_dir: str
        The directory that should (or will) contain the preprocessing models.

    Returns
    -------
    None
        If the models are already installed, nothing will happen. Otherwise, they will be installed.
        If the installation process encounters an error, this function prints the error to the console
        and quits Python.

    """
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(f"{models_dir}/synthmorph.affine.2.h5"):
        command = f"wget -O {models_dir}/synthmorph.affine.2.h5 https://surfer.nmr.mgh.harvard.edu/docs/synthmorph/synthmorph.affine.2.h5"
        print(command)
        result = run(command, shell=True)

        try:
            result.check_returncode()

        except Exception as error:
            print(f"synthmorph.affine.2.h5 could not be downloaded due to {error}")
            quit()

    if not os.path.exists(f"{models_dir}/synthmorph.affine.crop.h5"):
        command = f"wget -O {models_dir}/synthmorph.affine.crop.h5 https://surfer.nmr.mgh.harvard.edu/docs/synthmorph/synthmorph.affine.crop.h5"
        print(command)
        result = run(command, shell=True)

        try:
            result.check_returncode()

        except Exception as error:
            print(f"synthmorph.affine.crop.h5 could not be downloaded due to {error}")
            quit()


    if not os.path.exists(f"{models_dir}/synthmorph.deform.3.h5"):
        command = f"wget -O {models_dir}/synthmorph.deform.3.h5 https://surfer.nmr.mgh.harvard.edu/docs/synthmorph/synthmorph.deform.3.h5"
        print(command)
        result = run(command, shell=True)

        try:
            result.check_returncode()

        except Exception as error:
            print(f"synthmorph.deform.2.h5 could not be downloaded due to {error}")
            quit()

    if not os.path.exists(f"{models_dir}/synthmorph.rigid.1.h5"):
        command = f"wget -O {models_dir}/synthmorph.rigid.1.h5 https://surfer.nmr.mgh.harvard.edu/docs/synthmorph/synthmorph.rigid.1.h5"
        print(command)
        result = run(command, shell=True)

        try:
            result.check_returncode()

        except Exception as error:
            print(f"synthmorph.rigid.1.h5 could not be downloaded due to {error}")
            quit()

    if not os.path.exists(f"{models_dir}/synthstrip.1.pt"):
        command = f"wget -O {models_dir}/synthstrip.1.pt https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.1.pt"
        print(command)
        result = run(command, shell=True)

        try:
            result.check_returncode()

        except Exception as error:
            print(f"synthstrip.1.pt could not be downloaded due to {error}")
            quit()

    if not os.path.exists(f"{models_dir}/synthstrip.nocsf.1.pt"):
        command = f"wget -O {models_dir}/synthstrip.nocsf.1.pt https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.nocsf.1.pt"
        print(command)
        result = run(command, shell=True)

        try:
            result.check_returncode()

        except Exception as error:
            print(f"synthstrip.nocsf.1.pt could not be downloaded due to {error}")
            quit()

    if not os.path.exists(f"{models_dir}/synthstrip.infant.1.pt"):
        command = f"wget -O {models_dir}/synthstrip.infant.1.pt https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.infant.1.pt"
        print(command)
        result = run(command, shell=True)

        try:
            result.check_returncode()

        except Exception as error:
            print(f"synthstrip.infant.1.pt could not be downloaded due to {error}")
            quit()


def cpu_adjust(
    max_process_mem: int | float,
    cpus: int,
    threshold: float = 0.8
) -> int:
    """
    Confirms that the number of parallel processes will fit into the current amount of allowed
    memory and adjusts to an appropriate number if necessary.

    Parameters
    ----------
    max_process_mem: int | float
        The expected maximum memory (expressed in bytes) that will be consumed by each process.

    cpus: int
        The number of cpus used for multiprocessing.

    threshold: float
        The proportion of the available memory that can be used for the task. Defaults to 0.8.

    Returns
    -------
    cpus: int
        A potentially adjusted number of cpus that should fit into memory for a given task.

    Warnings
    --------
    UserWarning
        A warning is raised notifying the user that too many cpus were requested for a task
        and that the allowed amount has been lowered to a new value.
    """

    allowed_mem = int(psutil.virtual_memory().available * threshold)

    max_cpus = int(allowed_mem / max_process_mem)

    if cpus <= max_cpus:
        return cpus

    warnings.warn(
        f"There is not enough available memory to use {cpus} cpus for this task. "
        f"This task will proceed using {max_cpus} cpus.",
        UserWarning
    )
    return max_cpus


def update_errorfile(
    func_name: str,
    kwargs: Dict[str, Any],
    errorfile: Path | str,
    error: Exception
) -> None:
    """
    Updates an errorfile with the full traceback of an encountered exception and
    records the corresponding function and arguments.

    Parameters
    ----------
    func_name: str
        The name of the function that encountered an exception.

    kwargs: Dict[str, Any]
        The key word arguments used when executing the function.

    errorfile: Path | str
        The file in which errors will be recorded.

    error: Exception
        The exception that was encountered.

    """
    full_trace = "".join(
        traceback.TracebackException.from_exception(error).format()
    )

    args = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    log = f"{func_name}({args}) encountered the following exception:\n{full_trace}"

    errorfile = Path(errorfile)

    if errorfile.exists():
        log = f"\n\n\n\n{log}"

    with errorfile.open("a") as ef:
        ef.write(log)


def parse_string(s: str, pattern: str) -> Dict[str, str]:
    """
    Extracts variables from a string following a consistent pattern indicated using
    '{}'. Returns a dictionary with keys corresponding to the variable names.

    Parameters
    ----------
    s: str
        The string from which to extract variables

    pattern: str
        The string defining the search pattern. Variable names are encoded using '{}'
        (e.g. `pattern`='{patient}_{study}_{series}') would find values for the
        `patient`, `study`, and `series` variables.

    Returns
    -------
    Dict[str, str]
        A dictionary mapping variables provided in `pattern` to the values obtained
        from 's'.

    Raises
    ------
    ValueError
        An exception is raised if there are no variables encoded in `pattern` or if
        `s` is not in the format assumed by `pattern`.
    """
    variables = re.findall(r"\{(\w+)\}", pattern)

    if len(variables) == 0:
        raise ValueError(
            "pattern must contain at least one placeholder "
            "(e.g. pattern='{patient}_{study}_{series}') "
            f"but received pattern='{pattern}'."
        )

    regex = pattern

    for v in variables:
        regex = regex.replace(f"{{{v}}}", f"(?P<{v}>[^{{}}]+)")

    regex = re.compile(regex)

    match = regex.match(s)

    if match:
        return {v: match.group(v) for v in variables}

    else:
        raise ValueError(f"s='{s}' does not match the provided pattern='{pattern}'.")



def queue_batch(
    subdir: Path, queue: Queue, pattern: str = "*", batch_size: int | None = None
):
    """
    Processes a subdir by recursively globbing and placing batches of files into the queue.
    """
    generator = subdir.glob(f"**/{pattern}")

    if batch_size is not None:
        generator = iter(generator)

        while True:
            batch = list(islice(generator, batch_size))

            if not batch:
                break

            queue.put(batch)

    else:
        for file in generator:
            queue.put(file)


def cglob(
    root: Path | str = Path("."),
    pattern: str = "*",
    batch_size: int | None = None,
    queue_size: int = 20,
    cpus: int = 6,
):
    """
    Uses multiprocessing to parallelize the processing of top-level subdirectories,
    each handled by a separate process, while recursively globbing inside each subdirectory.
    """
    root = Path(root)
    cpus = max(cpus, 1)
    sub_dirs = list(root.glob("*/"))
    files = list(root.glob(pattern))

    if pattern != "*":
        batch = list(set(files))

    else:
        batch = list(set(files + sub_dirs))

    if batch_size is None:
        for file in batch:
            yield file

    else:
        yield batch

    queue = Queue(queue_size)
    processes = []
    remaining_subdirs = sub_dirs.copy()
    while remaining_subdirs or processes:
        while remaining_subdirs and len(processes) < cpus:
            subdir = remaining_subdirs.pop(0)
            p = Process(target=queue_batch, args=(subdir, queue, pattern, batch_size))
            p.start()
            processes.append(p)

        try:
            batch = queue.get_nowait()
            if batch is not None:
                yield batch

        except Exception:
            pass

        processes = [p for p in processes if p.is_alive()]

    for p in processes:
        p.join()


__all__ = [
    "MissingColumnsError",
    "MissingSoftwareError",
    "check_required_columns",
    "source_external_software",
    "sitk_to_surfa",
    "surfa_to_sitk",
    "initialize_models",
    "check_for_models",
    "cpu_adjust",
    "update_errorfile",
    "parse_string",
    "cglob"
]
