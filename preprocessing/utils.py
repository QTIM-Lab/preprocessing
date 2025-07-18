"""
The `utils` module contains custom exceptions and useful functions that are referenced
frequently throughout the rest of the library.

Public Classes
--------------
MissingColumnsError
    Exception to be raised in the cases where a DataFrame doesn't have the necessary
    columns to run a function.

Public Functions
----------------
check_required_columns
    Checks DataFrames for the presence of required columns.

sitk_to_surfa
    Convert a SimpleITK.Image to a surfa.Volume.

surfa_to_sitk
    Convert a surfa.Volume to a SimpleITK.Image.

hd_to_sitk
    Convert a highdicom.volume.Volume to a SimpleITK.Image.

sitk_to_hd
    Convert a SimpleITK.Image to a highdicom.volume.Volume.

hd_to_surfa
    Convert a hd.volume.Volume to a surfa.Volume.

surfa_to_hd
    Convert a hd.volume.Volume to a surfa.Volume.

niftiseg_to_dicomseg
    Convert a segmentation in NIfTI format to DICOM-SEG format.

dicomseg_to_niftiseg
    Convert a segmentation in DICOM-SEG format to NIfTI format.

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
import highdicom as hd

from typing import Sequence, Dict, Any, Literal
from SimpleITK import (
    Image,
    GetImageFromArray,
    GetArrayFromImage,
    ReadImage,
    WriteImage,
    Resample,
    sitkNearestNeighbor
)
from surfa import Volume, ImageGeometry
from subprocess import run
from pathlib import Path
from multiprocessing import Process, Queue
from itertools import islice
from pydicom import dcmread
from pydicom.sr.codedict import codes

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

        required_columns: Sequence[str]
            A sequence of the names of all of the required columns for a given function.

        optional_columns: Sequence[str]
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
        * spacing
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
    data = sf_im.data.transpose(2, 1, 0)

    if data.dtype == np.bool_:
        data = data.astype(int)

    spacing = sf_im.geom.voxsize

    sitk_im = GetImageFromArray(data)

    sitk_im.SetSpacing(spacing)

    sitk_im.SetDirection(
        (ras_lps @ sf_im.geom.vox2world[:3, :3] / spacing).flatten()
    )

    sitk_im.SetOrigin(ras_lps @ sf_im.geom.vox2world[:3, 3])

    return sitk_im


def hd_to_sitk(hd_im: hd.volume.Volume) -> Image:
    """
    Convert a highdicom.volume.Volume to a SimpleITK.Image.

    Parameters
    ----------
    hd_im: hd.volume.Volume
        A highdicom.volume.Volume to convert.

    Returns
    -------
    sitk_im: Image
        The corresponding SimpleITK.Image.
    """
    data = hd_im.array.transpose(2, 1, 0)

    if data.dtype == np.bool_:
        data = data.astype(int)

    sitk_im = GetImageFromArray(data)
    sitk_im.SetSpacing(hd_im.spacing)
    sitk_im.SetDirection(hd_im.direction.flatten())
    sitk_im.SetOrigin(hd_im.position)

    return sitk_im

def sitk_to_hd(sitk_im: Image) -> hd.volume.Volume:
    """
    Convert a SimpleITK.Image to a highdicom.volume.Volume.

    Parameters
    ----------
    sitk_im: Image
        A SimpleITK.Image to convert.

    Returns
    -------
    hd_im: hd.volume.Volume
        The corresponding highdicom.volume.Volume.
    """
    data = GetArrayFromImage(sitk_im).transpose(2, 1, 0)

    if data.dtype == np.bool_:
        data = data.astype(int)

    return hd.volume.Volume.from_components(
        data,
        spacing=sitk_im.GetSpacing(),
        coordinate_system="PATIENT",
        direction=np.reshape(sitk_im.GetDirection(), (3, 3)),
        position=sitk_im.GetOrigin()
    )

def hd_to_surfa(hd_im: hd.volume.Volume) -> Volume:
    """
    Convert a hd.volume.Volume to a surfa.Volume.

    Parameters
    ----------
    hd_im: hd.volume.Volume
        A hd.volume.Volume to convert.

    Returns
    -------
    sf_im: Volume
        The corresponding surfa.Volume.
    """
    lps_ras = np.diag([-1, -1, 1])

    vox2world = np.eye(4)
    vox2world[:3, :3] = (lps_ras @ hd_im.affine[:3, :3])
    vox2world[:3, 3] = lps_ras @ hd_im.affine[:3, 3]

    geom = ImageGeometry(shape=hd_im.shape, voxsize=hd_im.spacing, vox2world=vox2world)

    return Volume(hd_im.array, geom)

def surfa_to_hd(sf_im: Volume, dtype: np.dtype = np.float32) -> hd.volume.Volume:
    """
    Convert a hd.volume.Volume to a surfa.Volume.

    Parameters
    ----------
    hd_im: hd.volume.Volume
        A hd.volume.Volume to convert.

    Returns
    -------
    sf_im: Volume
        The corresponding surfa.Volume.
    """
    lps_ras = np.diag([-1, -1, 1])

    return hd.volume.Volume.from_components(
        sf_im.data.astype(dtype),
        spacing=sf_im.geom.voxsize,
        coordinate_system="PATIENT",
        position=lps_ras @ sf_im.geom.vox2world[:3, 3],
        direction=lps_ras @ sf_im.geom.rotation
    )

def niftiseg_to_dicomseg(
    dicom_dir: Path,
    nifti_seg: Path,
    dicom_seg: Path,
    model: Literal["meningioma", "glioma"] = "meningioma",
    tolerance: float = 0.05
):
    """
    Convert a segmentation in NIfTI format to DICOM-SEG format.

    Parameters
    ----------
    dicom_dir: Path
        Filepath to the reference DICOM series directory.

    nifti_seg: Path
        Filepath to the segmentation in NIfTI format.

    dicom_seg: Path
        Filepath to the output segmentation in DICOM-SEG format.

    model: str
        QTIM Lab segmentation model used to generate the segmentation. Choices include "meningioma" or "glioma".

    tolerance: float
        The conversion tolerance for `highdicom`'s Volume construction. Defaults to 0.05.

    Returns
    -------
    None
        A file is written to `dicom_seg` but nothing is returned.
    """
    dcms = hd.spatial.sort_datasets([dcmread(dcm) for dcm in dicom_dir.glob("**/*.dcm")])

    hd_im = hd.image.get_volume_from_series(dcms, atol=tolerance)
    sitk_im = hd_to_sitk(hd_im)
    sitk_seg = ReadImage(nifti_seg)

    sitk_seg_resampled = Resample(sitk_seg, sitk_im, interpolator=sitkNearestNeighbor)
    hd_seg = sitk_to_hd(sitk_seg_resampled)

    if model == "meningioma":
        algorithm_identification = hd.AlgorithmIdentificationSequence(
            name='QTIM Meningioma Segmenter',
            version='v1.0',
            family=codes.cid7162.ArtificialIntelligence
        )

        description_segment_1 = hd.seg.SegmentDescription(
            segment_number=1,
            segment_label='Meningioma - Enhancing Tumor',
            segmented_property_category=codes.SCT.MorphologicallyAbnormalStructure,
            segmented_property_type=codes.SCT.Neoplasm,
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
            algorithm_identification=algorithm_identification,
            tracking_uid=hd.UID(),
            tracking_id='Meningioma'
        )

        segmentation = hd.seg.Segmentation(
            source_images=dcms,
            pixel_array=hd_seg.array,
            segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
            segment_descriptions=[description_segment_1],
            series_instance_uid=hd.UID(),
            series_number=201,
            sop_instance_uid=hd.UID(),
            instance_number=1,
            manufacturer='Massachusetts General Hospital, QTIM Lab',
            manufacturer_model_name='QTIM Meningioma Segmenter v1.0',
            software_versions='v1.0',
            device_serial_number='QTIM Meningioma Segmenter',
            series_description="QTIM Meningioma Segmentation",
            content_label="MENINGIOMA"
        )

    elif model == "glioma":
        algorithm_identification = hd.AlgorithmIdentificationSequence(
            name='QTIM Glioma Segmenter',
            version='v1.0',
            family=codes.cid7162.ArtificialIntelligence
        )

        description_segment_1 = hd.seg.SegmentDescription(
            segment_number=1,
            segment_label='Glioma - Non-enhancing Tumor Core',
            segmented_property_category=codes.SCT.MorphologicallyAbnormalStructure,
            segmented_property_type=codes.SCT.Necrosis,
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
            algorithm_identification=algorithm_identification,
            tracking_uid=hd.UID(),
            tracking_id='Glioma'
        )

        description_segment_2 = hd.seg.SegmentDescription(
            segment_number=2,
            segment_label='Glioma - Surrounding Non-enhancing FLAIR Hyperintensity',
            segmented_property_category=codes.SCT.MorphologicallyAbnormalStructure,
            segmented_property_type=codes.SCT.Edema,
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
            algorithm_identification=algorithm_identification,
            tracking_uid=hd.UID(),
            tracking_id='Glioma'
        )

        description_segment_3 = hd.seg.SegmentDescription(
            segment_number=3,
            segment_label='Glioma - Enhancing Tumor',
            segmented_property_category=codes.SCT.MorphologicallyAbnormalStructure,
            segmented_property_type=codes.SCT.Neoplasm,
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
            algorithm_identification=algorithm_identification,
            tracking_uid=hd.UID(),
            tracking_id='Glioma'
        )

        description_segment_4 = hd.seg.SegmentDescription(
            segment_number=4,
            segment_label='Glioma - Resection Cavity',
            segmented_property_category=codes.SCT.MorphologicallyAbnormalStructure,
            segmented_property_type=hd.content.CodedConcept("63130001", "SCT", "Surgical Scar"),
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
            algorithm_identification=algorithm_identification,
            tracking_uid=hd.UID(),
            tracking_id='Glioma'
        )

        segmentation = hd.seg.Segmentation(
            source_images=dcms,
            pixel_array=hd_seg.array,
            segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
            segment_descriptions=[
                description_segment_1,
                description_segment_2,
                description_segment_3,
                description_segment_4
            ],
            series_instance_uid=hd.UID(),
            series_number=301,
            sop_instance_uid=hd.UID(),
            instance_number=1,
            manufacturer='Massachusetts General Hospital, QTIM Lab',
            manufacturer_model_name='QTIM Glioma Segmenter v1.0',
            software_versions='v1.0',
            device_serial_number='QTIM Glioma Segmenter',
            series_description="QTIM Glioma Segmentation",
            content_label="GLIOMA"
        )

    dicom_seg.parent.mkdir(parents=True, exist_ok=True)
    segmentation.save_as(dicom_seg)

def dicomseg_to_niftiseg(
    dicom_seg: Path,
    nifti_seg: Path,
):
    """
    Convert a segmentation in DICOM-SEG format to NIfTI format.

    Parameters
    ----------
    dicom_seg: Path
        Filepath to the segmentation in DICOM-SEG format.

    nifti_seg: Path
        Filepath to the output segmentation in NIfTI format.

    Returns
    -------
    None
        A file is written to `nifti_seg` but nothing is returned.
    """
    hd_seg = hd.seg.segread(dicom_seg).get_volume()
    sitk_seg = hd_to_sitk(hd_seg)

    nifti_seg.parent.mkdir(parents=True, exist_ok=True)
    WriteImage(sitk_seg, nifti_seg)

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
which is not currently defined. If you are on a QTIM Machine, please enter 'QTIM' below.
Otherwise, enter the path where you wish to store the Synthstrip and Synthmorph models
(this directory will be created if it does not yet exist):\n
"""
    )

    if models_dir.lower() == "qtim":
        models_dir = "/autofs/space/crater_001/tools/preprocessing_models"

    add_to_rc = input(
        f"\nWould you like to add 'export PREPROCESSING_MODELS_PATH={models_dir}' to your RC file? (y/N):\n\n"
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
    Can be disabled if $SUPPRESS_MODEL_DOWNLOADS is set.

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
    if eval(os.environ.get("SUPPRESS_MODEL_DOWNLOADS", "False")):
        return
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
    error: Exception,
    verbose: bool = False,
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

    verbose: bool
        Whether to print the full traceback in addition to logging it
        to `errorfile`. Defaults to False.
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

    if verbose:
        print(log)


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
        (e.g. `pattern`='{patient}_{study}_{series}' would find values for the
        `patient`, `study`, and `series` variables).

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
    Processes a subdir by recursively 'globbing' and placing batches of files into the queue.

    Parameters
    ----------
    subdir: Path
        The subdirectory to be 'globbed'.

    queue: Queue
        The queue used to store files or batches of files matching the search pattern.

    pattern: str
        The search pattern used to select files ('**/' already applied).

    batch_size: int | None
        The length of the sequences of files placed in the queue. If `None`, single files
        will be stored.
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
    each handled by a separate process, while recursively 'globbing' inside each subdirectory.

    Parameters
    ----------
    root: Path | str
        The root directory to be 'globbed', assumed to have subdirectories.

    pattern: str
        The search pattern used to select files ('**/' already applied).

    batch_size: int | None
        The length of the sequences of files placed in the queue. If `None`, single files
        will be stored.

    queue_size: int
        The maximum size of the queue in terms of numbers of batches.

    cpus: int
        Number of cpus to use for multiprocessing. Defaults to 1 (no multiprocessing).

    Yields
    ------
    Sequence[Path] | Path
        This function yields a file matching the search pattern within `root` or a batch
        of such files dependent on `batch_size`.
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
    "check_required_columns",
    "sitk_to_surfa",
    "surfa_to_sitk",
    "hd_to_sitk",
    "sitk_to_hd",
    "hd_to_surfa",
    "surfa_to_hd",
    "niftiseg_to_dicomseg",
    "dicomseg_to_niftiseg",
    "initialize_models",
    "check_for_models",
    "cpu_adjust",
    "update_errorfile",
    "parse_string",
    "cglob"
]
