"""Tools for converting between image formats."""
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from highdicom.base import SOPClass

import SimpleITK as sitk

import numpy as np

import pydicom


from pydicom.uid import RLELossless
from pydicom.valuerep import format_number_as_ds
import highdicom as hd

import os


from pydicom.uid import (
    ImplicitVRLittleEndian,
    ExplicitVRLittleEndian,
    RLELossless,
    JPEGBaseline8Bit,
    JPEG2000Lossless,
    JPEGLSLossless,
)

from pydicom.encaps import encapsulate

from highdicom.enum import (
    AnatomicalOrientationTypeValues,
    CoordinateSystemNames,
    PhotometricInterpretationValues,
    LateralityValues,
    PatientOrientationValuesBiped,
    PatientOrientationValuesQuadruped,
    PatientSexValues,
)
from highdicom.frame import encode_frame

pydicom.config.datetime_conversion = True


"""curie.imaging.processing.utils.sitk General purpose SimpleITK utilities."""

# Standard axis-aligned SimpleITK direction matrices for the three viewing planes
STANDARD_DIRECTION_MATRICES = {
    "axial": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    "sagittal": (0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
    "coronal": (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0),
}


def get_center(im: sitk.Image) -> Tuple[float, ...]:
    """Get the center of a SimpleITK image in physical space.

    Parameters
    ----------
    im: sitk.Image
        A SimpleITK image. May be 2D or 3D.

    Returns
    -------
    center: Tuple[float, ...]
        The center of the image in physical space

    """
    center_index = [(s - 1) / 2 for s in im.GetSize()]
    return im.TransformContinuousIndexToPhysicalPoint(center_index)


def direction_is_reversed(dir_matrix: Sequence[float]) -> bool:
    """Check whether a SimpleITK direction matrix is reversed.

    A reversed direction matrix is one in which the third direction is in the opposite
    direction to the cross-product of the first two directions.

    Parameters
    ----------
    dir_matrix: Sequence[float]
        Direction matrix in the standard sitk format (9 elements in a 1D Sequence).

    Returns
    -------
    bool:
        True if and only if the direction matrix is reversed.

    """
    m = np.array(dir_matrix).reshape((3, 3))
    d0 = m[:, 0]
    d1 = m[:, 1]
    d2 = m[:, 2]

    d2_assumed = np.cross(d0, d1)

    return bool(np.dot(d2, d2_assumed) < 0.0)


"""curie.imaging.dicom.utils Dicom utils."""


CCDS_AUTH_TOKEN_URL = "https://api.ccds.io/auth/token"
CCDS_MLPROJ_DICOMWEB_URL = "https://mlproj.ccds.io/{}/ims/dicom-web"


def compute_anatomical_plane(image_orient_patient: List[float]) -> str:
    """Compute the anatomical plane from the populated series plan orientation (from ImageOrientationPatient dcm tag).

    Parameters
    ----------
    image_orient_patient:
        ImagePatientOrientation dicom tag

    Returns
    -------
        either 'sagittal', 'coronal' or 'axial'

    """
    iop = [x for x in image_orient_patient]
    plane = np.cross(iop[0:3], iop[3:6])
    plane = [abs(x) for x in plane]
    direction = np.argmax(plane).item()
    direction_name = ["sagittal", "coronal", "axial"]
    try:
        return direction_name[int(direction)]
    except IndexError:
        raise ValueError("Invalid direction value: {}".format(direction))


def check_equal_lists_of_nb(
    list_of: List[List[Union[int, float]]], nb_decimals: int = 3
) -> bool:
    """Check if the elements of a list are all equal.

    Parameters
    ----------
    list_of:
        list to check

    Returns
    -------
        True if elements of a list are all equal

    """
    if len(list_of) == 0:
        return False
    elif len(list_of) <= 1:
        return True

    list_of_set = []
    for list_ in list_of:
        # round values to avoid precision issues
        list_ = [np.round(float(el), nb_decimals) for el in list_]
        list_of_set.append(set(list_))
    # at least two elements are not equal
    return all(x == list_of_set[0] for x in list_of_set[1:])


def list_dcm_folder(dcm_folder: str) -> List[str]:
    """List all dicom files in a DICOM folder recursively.

    Parameters
    ----------
    dcm_folder: str
        path of the DICOM folder to be listed.

    Returns
    -------
    List[str]
        list of DICOM files

    """
    list_dcm = []
    for root, directories, filenames in os.walk(dcm_folder):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            try:
                pydicom.dcmread(
                    filepath,
                    stop_before_pixels=True,
                    specific_tags=["SeriesInstanceUID"],
                )
                list_dcm.append(filepath)
            except pydicom.errors.InvalidDicomError:
                continue
    return list_dcm


"""curie.imaging.dicom.order Tools for DICOM slice ordering."""


def calc_slice_distance(
    image_orientation_patient: Sequence[Union[float, str]],
    image_position_patient: Sequence[Union[float, str]],
) -> float:
    """Calculate the signed distance of a slice in 3D space from the origin of the frame of reference.

    Parameters
    ----------
    image_orientation_patient:
        value of DICOM attribute *ImageOrientationPatient*
    image_position_patient:
        value of DICOM attribute *ImagePositionPatient*

    Returns
    -------
    float
        signed distance

    """
    orientation = np.array(image_orientation_patient, dtype=float)
    position = np.array(image_position_patient, dtype=float)

    if len(orientation) != 6 or len(position) != 3:
        raise ValueError(
            "Wrong number of value for image_orientation_patient or image_position_patient"
        )

    normal = np.cross(orientation[0:3], orientation[3:6])
    return float(np.dot(normal, position))


def sort_slices(
    datasets: Sequence[pydicom.Dataset], group_by_position=False
) -> Sequence[pydicom.Dataset]:
    """Sort DICOM datasets of single-frame image instances.

    Sorting is based on the values of the ImageOrientationPatient,
    ImagePositionPatient, and InstanceNumber.

    Parameters
    ----------
    datasets:
        DICOM data sets of single-frame image instances

    Returns
    -------
        sorted DICOM data sets

    """

    def sort_func(ds: pydicom.Dataset) -> Tuple[float, int]:
        try:
            distance = calc_slice_distance(
                ds.ImageOrientationPatient, ds.ImagePositionPatient
            )
            return (distance, int(ds.InstanceNumber))
        except AttributeError:
            if not hasattr(ds, "InstanceNumber"):
                raise AttributeError(
                    "dataset should have a least InstanceNumber or \
                                                    (ImageOrientationPatient, ImagePositionPatient)"
                )
            return (0, int(ds.InstanceNumber))

    if len(datasets) == 0:
        return []

    try:
        presorted_datasets = sorted(datasets, key=sort_func, reverse=False)
        sorted_datasets = []
        distances = [sort_func(ds)[0] for ds in datasets]
        unique_distances = set(distances)
        n = int(len(distances) / len(unique_distances))

        if group_by_position:
            for i in range(n):
                sub_list = []
                for ds in presorted_datasets[i::n]:
                    sub_list.append(ds)
                sorted_datasets.append(sub_list)
        else:
            for i in range(n):
                for ds in presorted_datasets[i::n]:
                    sorted_datasets.append(ds)

        return sorted_datasets
    except AttributeError as e:
        print("Unable to sort datasets: ", str(e))
        return datasets


def get_datasets_spacing(datasets: Sequence[pydicom.Dataset]) -> np.ndarray:
    """Compute pixel spacing along the 3 dimensions of the voxel array.

    Choose the most common distance value between slices.

    Returns the spacing in SimpleITK convention, in which the first two
    spacings are swapped with respect to the order found in the DICOM
    PixelSpacing tag.

    Parameters
    ----------
    datasets: List[pydicom.dataset.Dataset]
        List of DICOM datasets of single-frame image SOP instances that form a
        series.

    Returns
    -------
    np.ndarray
        3-dimensional pixel spacing.

    Notes
    -----
    Expects the input list of ``pydicom.Dataset`` to be sorted.

    """
    if len(datasets) == 0:
        raise ValueError("The input list of `pydicom.Dataset` is empty.")
    if len(datasets) > 1:
        distances = []
        for ds in datasets:
            distance = calc_slice_distance(
                ds.ImageOrientationPatient, ds.ImagePositionPatient
            )
            distances.append(distance)
        differences = np.diff(distances)
        values, counts = np.unique(differences, return_counts=True)
        dist = values[counts == counts.max()][0]
        # NB swap spacing in first two directions due to differing conventions
        spacing = np.array(
            [
                float(datasets[0].PixelSpacing[1]),
                float(datasets[0].PixelSpacing[0]),
                dist,
            ],
            float,
        )
    elif len(datasets) == 1:
        # NB swap spacing in first two directions due to differing conventions
        spacing = np.array(
            [float(datasets[0].PixelSpacing[1]), float(datasets[0].PixelSpacing[0]), 1],
            float,
        )
    return spacing


def sitk_direction_to_dicom_iop(
    direction: Union[np.ndarray, Sequence[float]]
) -> List[float]:
    """Translate an SimpleITK direction matrix to a DICOM image orientation (patient) tag.

    Returns an array suitable to place in the ImageOrientationPatient tag
    (0020,0037) of a DICOM file. This tag encodes the direction of the
    axes of the image in physical space.

    Parameters
    ----------
    direction: Union[np.ndarray, Sequence[float]]
        Direction matrix in the SimpleITK format for a 3D image
        (consisting of 9 numbers in a 1D array). For example, as
        obtained from the ``SimpleITK.Image.GetDirection()`` method on
        a SimpleITK Image object.

    Returns
    -------
    List[float]
        Direction cosines in the format used in the DICOM
        ImageOrientationPatient tag.

    Raises
    ------
    ValueError
        If the input cannot be converted to a numpy array with 9 elements.

    Notes
    -----
        The direction of the third axis is lost in this conversion.
        Consequently using:

        >>> dicom_iop_to_sitk_direction(sitk_direction_to_dicom_iop(im.GetDirection()))

        may result in a direction matrix that has the third axis flipped compared
        to the original image.

    """
    try:
        dir_array = np.array(direction)
    except Exception:
        raise ValueError("Input type cannot be converted to a numpy array")

    if dir_array.shape != (9,):
        raise ValueError("Input array must have 9 elements")

    # Pick out the correct elements of the array to give the directions of the
    # first two image axes
    return list(dir_array[[0, 3, 6, 1, 4, 7]])


def dicom_iop_to_sitk_direction(
    iop: Union[np.ndarray, Sequence[float]]
) -> Tuple[float, ...]:
    """Translate a DICOM ImageOrientationPatient tag into an SimpleITK direction matrix.

    The DICOM ImageOrientationPatient (0020,0037) tag defines the direction cosines of the
    two in-plane image directions, whereas the SimpleITK uses a 9 element direction matrix
    that describes the directions of all three directions.

    Note that this conversion assumes that the third image axis, whose direction is
    not explicitly defined within a DICOM dataset, is orthogonal to the
    first and second image axes.

    Parameters
    ----------
    iop: Union[np.ndarray, Sequence[float]]
        Array of 6 floats in the format of a DICOM ImageOrientationPatient
        tag. For example as a result of calling ``dcm.ImageOrientationPatient``
        on a ``pydicom.Dataset``.

    Returns
    -------
    Tuple[float, ...]
        Direction matrix in the flattened format used by SimpleITK.

    Raises
    ------
    ValueError
        If the input sequence does not contain the correct number of elements

    """
    if len(iop) != 6:
        raise ValueError("Input array should have length 6")

    # Get the first two (in-plane) unit vectors
    d1 = np.array(iop[:3])
    d2 = np.array(iop[3:])

    # Calculate the third (out-of-plane) direction as the cross-product of the
    # first two
    d3 = np.cross(d1, d2)
    dir_matrix = np.stack([d1, d2, d3], axis=1)

    return tuple(float(x) for x in dir_matrix.flatten())


def dcm_series_to_sitk_image(
    datasets: Sequence[pydicom.Dataset],
    rescale: bool = True,
    dtype: Optional[np.dtype] = None,
    order_datasets: Optional[bool] = True,
) -> sitk.Image:
    """Create a SimpleITK image from a list of single-frame pydicom datasets.

    Parameters
    ----------
    datasets: List[pydicom.dataset.Dataset]
        List of DICOM datasets of single-frame image SOP instances that form a
        series.
    rescale: bool
        whether pixel values should be rescaled by applying the values of the
        ``RescaleSlope`` and ``RescaleIntercept`` attributes.
        Default: ``True``.
    dtype: np.dtype, optional
        A numpy datatype used during the conversion process. By default, if ``'rescale'``
        is ``True``, ``float`` will be used, whereas if ``'rescale'`` is ``False``,
        an appropriate integer type will be chosen.
    order_datasets: bool, optional
        If ``True``, order the input pydicom dataset spatially.
        Default: ``True``.

    Returns
    -------
    SimpleITK.Image
        3-dimensional SimpleITK image object.

    Raises
    ------
    ValueError
        If the input datasets list is empty.
    AttributeError, IndexError
        If either the ``RescaleSlope`` or ``RescaleIntercept`` is not specified or
        incorrectly allocated in a ``pydicom.Dataset`` of a SOP instance.

    Notes
    -----
    There is no requirement that the elements of dataset be sorted, spatially
    or otherwise. Spatial ordering of the slices is performed automatically unless
    ``order_datasets`` is set to ``False``.

    """
    n = len(datasets)

    if n == 0:
        raise ValueError("The input list of `pydicom.Dataset` is empty.")

    if order_datasets:
        datasets = sort_slices(datasets)

    origin = np.array(datasets[0].ImagePositionPatient, float)
    direction = np.array(
        dicom_iop_to_sitk_direction(datasets[0].ImageOrientationPatient)
    )
    spacing = get_datasets_spacing(datasets)

    if rescale:
        # Rescale slope and intercept may be arbitrary floating point numbers, so to maintain fill fidelity
        # we must use floating point
        if dtype is None:
            dtype = np.dtype("float")
    else:
        if dtype is None:
            if datasets[0].Modality == "CT":
                # In some cases pydicom return pixel data as unsigned integers, which
                # creates problems with values in Hounsfield unit.
                dtype = np.dtype("int16")
            else:
                dtype = datasets[0].pixel_array.dtype
    array = np.zeros((n, datasets[0].Rows, datasets[0].Columns), dtype)

    for i, ds in enumerate(datasets):
        frame = ds.pixel_array.astype(dtype, copy=False)
        if rescale:
            try:
                slope = np.array([float(getattr(ds, "RescaleSlope", 1))], dtype).item()
                intercept = np.array(
                    [float(getattr(ds, "RescaleIntercept", 0))], dtype
                ).item()
            except (AttributeError, IndexError):
                slope = np.array([1], dtype).item()
                intercept = np.array([0], dtype).item()
            array[i, :, :] = slope * frame + intercept
        else:
            array[i, :, :] = frame

    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction)

    return image


def multiframe_dcm_to_sitk(
    dataset: pydicom.Dataset, rescale: bool = True, dtype: Optional[np.dtype] = None
) -> sitk.Image:
    """Create a SimpleITK image object from a multi-frame pydicom dataset.

    Parameters
    ----------
    dataset: pydicom.dataset.Dataset
        DICOM data set of multi-frame image SOP instance.
    rescale: bool, optional
        Whether pixel values should be rescaled by applying the values of the
        ``RescaleSlope`` and ``RescaleIntercept`` attributes.
        Default: ``True``.
    dtype: np.dtype, optional
        A numpy datatype used during the conversion process. By default, if ``'rescale'``
        is ``True``, ``float`` will be used, whereas if ``'rescale'`` is ``False``,
        an appropriate integer type will be chosen.

    Returns
    -------
    SimpleITK.Image
        3-dimensional SimpleITK image object.

    Raises
    ------
    ValueError
        If the dataset is not a multi-frame (enhanced) DICOM instance.
    AttributeError, IndexError
        If either the ``RescaleSlope`` or ``RescaleIntercept`` is not specified or
        incorrectly allocated in a ``pydicom.Dataset`` of a SOP instance.

    Notes
    -----
    There is no requirement that the elements of dataset be sorted, spatially
    or otherwise. Spatial ordering of the slices is performed automatically.

    """
    try:
        shared_func_groups = dataset.SharedFunctionalGroupsSequence[0]
    except (IndexError, AttributeError):
        raise ValueError("Dataset must represent a multi-frame image instance.")
    perframe_func_groups = dataset.PerFrameFunctionalGroupsSequence
    n = int(dataset.NumberOfFrames)

    # Direction cosines for the 1st and 2nd axis of the voxel array.
    direction = np.array(
        shared_func_groups.PlaneOrientationSequence[0].ImageOrientationPatient, float
    )

    # In the special case where n is one, the direction and spacing of the
    # third dimension are assumed to be a unit vector perpendicular to the
    # first two image axes
    if n == 1:
        direction = np.array(dicom_iop_to_sitk_direction(direction))
        origin = np.array(
            perframe_func_groups[0].PlanePositionSequence[0].ImagePositionPatient, float
        )
        spacing = np.array(
            [
                # NB swap spacing in first two directions due to differing conventions
                float(shared_func_groups.PixelMeasuresSequence[0].PixelSpacing[1]),
                float(shared_func_groups.PixelMeasuresSequence[0].PixelSpacing[0]),
                1.0,
            ],
            float,
        )
        indices = np.asarray([0])
    else:
        # Compute the direction cosines for the 3rd axis of voxel array.
        positions = np.zeros((n, 3), float)
        distances = np.zeros((n,), float)
        for i in range(n):
            frame = perframe_func_groups[i]
            pos = np.array(frame.PlanePositionSequence[0].ImagePositionPatient, float)
            positions[i, :] = pos
            distances[i] = calc_slice_distance(list(direction), list(pos))
        indices = np.argsort(distances)
        first_pos = positions[indices[0]]
        last_pos = positions[indices[-1]]
        v = last_pos - first_pos
        v = v / np.linalg.norm(v)
        direction = np.stack([direction[:3], direction[3:], v], axis=1).flatten()

        # Determine origin of frame of reference
        origin = np.array(first_pos, float)

        # Compute pixel spacing along the 3 dimensions of the voxel array.
        # Choose the most common distance value between slices.
        differences = np.diff(distances)
        values, counts = np.unique(differences, return_counts=True)
        dist = values[counts == counts.max()][0]
        spacing = np.array(
            [
                # NB swap spacing in first two directions due to differing conventions
                float(shared_func_groups.PixelMeasuresSequence[0].PixelSpacing[1]),
                float(shared_func_groups.PixelMeasuresSequence[0].PixelSpacing[0]),
                dist,
            ],
            float,
        )

    if rescale:
        # Rescale slope and intercept may be arbitrary floating point numbers, so to maintain fill fidelity
        # we must use floating point
        if dtype is None:
            dtype = np.dtype("float")
        array = np.zeros((n, dataset.Rows, dataset.Columns), dtype)
        for i in range(n):
            index = indices[i]
            frame = dataset.pixel_array[index].astype(dtype, copy=False)
            try:
                s = perframe_func_groups[i].PixelValueTransformationSequence[0]
                slope = np.array([float(getattr(s, "RescaleSlope", 1))], dtype).item()
                intercept = np.array(
                    [float(getattr(s, "RescaleIntercept", 0))], dtype
                ).item()
            except (AttributeError, IndexError):
                slope = np.array([1], dtype).item()
                intercept = np.array([0], dtype).item()
            array[i, :, :] = slope * frame + intercept
    else:
        if dtype is None:
            if dataset.Modality == "CT":
                # In some cases pydicom return pixel data as unsigned integers, which
                # creates problems with values in Hounsfield unit.
                dtype = np.dtype("int16")
            else:
                dtype = dataset.pixel_array.dtype
        # In the special case where n is one, pydicom interpret the pixel_array
        # with shape (ds.Rows, ds.Columns) instead of (n, ds.Rows, ds.Columns)
        if n == 1:
            array = dataset.pixel_array.astype(dtype, copy=False)
            array = np.expand_dims(array, axis=0)
        else:
            array = dataset.pixel_array[indices, :, :].astype(dtype, copy=False)

    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction)

    return image


def sitk_image_to_pydicom_series(
    im: sitk.Image,
    ref_dataset: pydicom.Dataset,
    series_instance_uid: Optional[str] = None,
    series_number: int = 100,
    series_description: Optional[str] = None,
    reverse_instances: Optional[bool] = False,
) -> List[pydicom.Dataset]:
    """Convert an SITK image to a series of pydicom datasets.

    The image is divided along the third dimension into slices, each of which is recorded as a single
    instance.

    Basic information is copied across from the reference dataset, but the user is responsible for
    ensuring that all the required metadata is populated in the resulting dataset.

    All geometric information is converted from the geometric information in the sitk image,
    including origin, spacing and direction. Slice thickness is assumed to the spacing in the
    third dimension of the sitk image.

    Parameters
    ----------
    im: sitk.Image
        The 3D SITK image to convert into pydicom datasets
    ref_dataset: pydicom.Dataset
        Reference dataset to copy basic study and patient information from.
    series_instance_uid: Optional[int]
        Series instance UID for the new series. If unspecified, a UID will be generated.
    series_number: int
        Number of the new series within the study. Default: 100.
    series_description: Optional[str]
        Series description for the new series.
    reverse_instances: Optional[bool]
        Write out the instances in the reverse order to that in which the
        slices appear down the third dimension in the SimpleITK image. Some
        viewers use instance number to determine scrolling direction, in which
        case this option can be used to ensure that the desired scrolling
        direction is obtained. A value of False means that instance numbers
        will match the slice index, a value of True means that the instance
        numbers will be in the opposite direction to the slice indices. A value
        of None means that the behavior will be determined automatically based
        on the anatomical view in order to match conventions. Note that the
        order of instances in the returned list will always be sorted by
        increasing instance number.

    Returns
    -------
    List[pydicom.Dataset]
        Series of pydicom datasets created from the input image, sorted by increasing instance number.

    Raises
    ------
    ValueError
        If the input image is not 3 dimensional

    """
    # Check image dimensionality
    if im.GetDimension() != 3:
        raise ValueError("Input image must be 3D")

    if series_instance_uid is None:
        series_instance_uid = pydicom.uid.generate_uid()

    if reverse_instances is None:
        # Automatically determine whether the instances should be reversed based on image orientation
        im_view = compute_anatomical_plane(
            sitk_direction_to_dicom_iop(im.GetDirection())
        )

        # Check whether the image is already reversed down the third dimension
        input_reversed = direction_is_reversed(im.GetDirection())

        # Axial and sagittal images are usually ordered in the opposite direction to the SimpleITK convention
        if im_view in ["axial", "sagittal"]:
            reverse_instances = not input_reversed
        else:
            reverse_instances = input_reversed

    if reverse_instances:
        # Flip the image so that the instances will be returned in reverse order
        im = sitk.Flip(im, [False, False, True])

    # Loop over slices of the last dimension of the input image
    datasets = []
    for i in range(im.GetSize()[2]):
        # Get this slice as an array
        slice_im = im[:, :, i : i + 1]
        slice_arr = np.squeeze(sitk.GetArrayFromImage(slice_im))

        # Translate the orientation into the DICOM tag convention
        orientation = sitk_direction_to_dicom_iop(slice_im.GetDirection())

        # Create a dataset for this slice
        slice_dataset = SOPClass(
            study_instance_uid=ref_dataset.StudyInstanceUID,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=pydicom.uid.generate_uid(),
            sop_class_uid=ref_dataset.SOPClassUID,
            instance_number=i,
            modality=ref_dataset.Modality,
            manufacturer=getattr(ref_dataset, "Manufacturer", None),
            patient_id=getattr(ref_dataset, "PatientID", None),
            patient_name=getattr(ref_dataset, "PatientName", None),
            patient_birth_date=getattr(ref_dataset, "PatientBirthDate", None),
            patient_sex=getattr(ref_dataset, "PatientSex", None),
            accession_number=getattr(ref_dataset, "AccessionNumber", None),
            study_id=getattr(ref_dataset, "StudyID", None),
            study_date=getattr(ref_dataset, "StudyDate", None),
            study_time=getattr(ref_dataset, "StudyTime", None),
            referring_physician_name=getattr(
                ref_dataset, "ReferringPhysicianName", None
            ),
            series_description=series_description,
        )

        # Instance Creation Time (now)
        slice_dataset.InstanceCreationTime = datetime.now().time()
        slice_dataset.InstanceCreationDate = datetime.today()

        # Geometrical information
        slice_dataset.ImageOrientationPatient = orientation
        slice_dataset.ImagePositionPatient = list(slice_im.GetOrigin())
        slice_dataset.PixelSpacing = list(slice_im.GetSpacing()[:2])
        slice_dataset.SliceThickness = slice_im.GetSpacing()[2]
        if hasattr(ref_dataset, "FrameOfReferenceUID"):
            slice_dataset.FrameOfReferenceUID = ref_dataset.FrameOfReferenceUID

        # Pixel interpretation information
        slice_dataset.Rows = slice_arr.shape[0]
        slice_dataset.Columns = slice_arr.shape[1]
        slice_dataset.SamplesPerPixel = 1
        slice_dataset.PixelRepresentation = 0
        slice_dataset.BitsAllocated = 16
        slice_dataset.BitsStored = 16
        slice_dataset.PhotometricInterpretation = "MONOCHROME2"

        # Pixels
        rescale_intercept = slice_arr.min()
        slice_dataset.RescaleIntercept = rescale_intercept
        slice_dataset.RescaleSlope = 1.0
        pixels = (slice_arr - rescale_intercept).astype(np.uint16).tobytes()
        pixel_data = pydicom.DataElement(
            tag=pydicom.datadict.tag_for_keyword("PixelData"), VR="OW", value=pixels
        )
        slice_dataset.add(pixel_data)

        datasets.append(slice_dataset)

    return datasets


MANUFACTURER = "MGB Center for Clinical Data Science"

# Keywords for attributes that should not be copied to new instances
OMIT_KEYWORDS = [
    "MilitaryRank",
    "BranchOfService",
    "StackID",
    "StudyStatusID",
    "StudyPriorityID",
    "RequestingService",
    "StudyComments",
    "PerformedStationName",
    "PerformedLocation",
    "RequestedProcedureID",
    "IssuerOfAdmissionID",
    "PixelData",
    "InstanceNumber",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "RequestedProcedureDescription",
    "CurrentPatientLocation",
    "ManufacturerModelName",
]


def make_display_sc(
    pixels: List[np.ndarray],
    instance_ds: pydicom.Dataset,
    copy_spatial_information: bool = False,
) -> List[hd.sc.SCImage]:
    """Create DICOM Secondary Capture output for the segmentation overlays.

    Parameters
    ----------
    pixels: List[np.ndarray]
        List of raw RGB pixel arrays. Each will create an instance of the secondary
        capture series
    instance_ds: pydicom.Dataset
        One dataset from the original series, from which patient/study information
        will be copied.
    copy_spatial_information: bool
        If True, spatial information (position, orientation) will
        also be copied from the instance_ds. Note that this is not part of the
        standard for SC images.

    Returns
    -------
    List[hd.sc.SCImage]
        Secondary capture datasets, one per input pixel array, constituting a new
        series

    """
    series_instance_uid = hd.UID()
    patient_orientation = (
        hd.enum.PatientOrientationValuesBiped.P,  # posterior
        hd.enum.PatientOrientationValuesBiped.F,  # foot
    )

    out = []
    now = datetime.now()

    for p, pixel_array in enumerate(pixels, 1):
        # Spacing may have changed with the plotting functions
        pixel_spacing = [
            instance_ds.PixelSpacing[0] * instance_ds.Rows / pixel_array.shape[0],
            instance_ds.PixelSpacing[1] * instance_ds.Columns / pixel_array.shape[1],
        ]

        sc = hd.sc.SCImage.from_ref_dataset(
            ref_dataset=instance_ds,
            pixel_array=pixel_array,
            photometric_interpretation="RGB",
            bits_allocated=8,
            coordinate_system=hd.enum.CoordinateSystemNames.PATIENT,
            series_instance_uid=series_instance_uid,
            series_number=100,
            sop_instance_uid=hd.UID(),
            instance_number=p,
            manufacturer=MANUFACTURER,
            patient_orientation=patient_orientation,
            series_description="DeepSpine Vertebral Segmentation",
            pixel_spacing=pixel_spacing,
            transfer_syntax_uid=RLELossless,
        )
        sc.ContentDate = now.strftime("%Y%m%d")
        sc.ContentTime = now.strftime("%H%M%S.%f")
        sc.SeriesDate = sc.ContentDate
        sc.SeriesTime = sc.ContentTime

        if copy_spatial_information:
            # Copy spatial information
            # This does not conform to the standard, but helps with spatial
            # cross-referencing in Imaging Fabric
            sc.ImageOrientationPatient = instance_ds.ImageOrientationPatient
            sc.ImagePositionPatient = instance_ds.ImagePositionPatient
            sc.FrameOfReferenceUID = instance_ds.FrameOfReferenceUID

        out.append(sc)

    return out


def get_shared_metadata(
    source_datasets: List[pydicom.Dataset],
    omit_keywords: Optional[Sequence[str]] = None,
) -> pydicom.Dataset:
    """Find metadata that is shared by all instances in a series

    Parameters
    ----------
    source_datasets: List[pydicom.Dataset]
        List of datasets from the original series
    omit_keywords: Optional[Sequence[str]]
        List of keywords that should not be copied to the output dataset

    Returns
    -------
    series_metadata: pydicom.Dataset
        Dataset containing all the elements that are shared between
        every instance in the original series. Private tags are ignored.

    """
    if omit_keywords is None:
        omit_keywords = []

    # Copy information from the reference datasets if the values are consistent
    # across all source instances
    series_metadata = pydicom.Dataset()
    for el in source_datasets[0].elements():
        # NB Some elements are encoded as RawDataElements, which have no keyword...
        kw = pydicom.datadict.keyword_for_tag(el.tag)
        if not el.tag.is_private:
            if kw not in omit_keywords:
                try:
                    if all(
                        (el.tag in ds and ds[el.tag] == source_datasets[0][el.tag])
                        for ds in source_datasets[1:]
                    ):
                        series_metadata.add(el)
                except NotImplementedError:
                    # This has been found to happen with tags with unknown VRs
                    continue

    return series_metadata


def make_mpr(
    images: List[Optional[sitk.Image]],
    source_datasets: List[pydicom.Dataset],
    series_description: str,
    series_number: int,
    transfer_syntax_uid: str,
) -> Dict[str, List[pydicom.Dataset]]:
    """Convert a single 3D SITK images into a pydicom series with metadata.

    Parameters
    ----------
    images: sitk.Image
        List of 3D SITK image to convert (as a single series). Entries may be
        None, corresponding to missing crops.
    source_datasets: List[pydicom.Dataset]
        Dataset from the source image as a pydicom objects to copy information from.
    series_description: str
        Series description to put into the DICOM files
    series_number: int
        Series number to use for the new series

    Returns
    -------
    Dict[str, List[pydicom.Dataset]]:
        Dictionary of output DICOM objects grouped by disc labels. Keys are
        disc labels, values are lists each containing a subset of pydicom
        datasets comprising the full MIPs series.

    """
    # Get metadata shared by elements in the series
    series_metadata = get_shared_metadata(
        source_datasets,
        omit_keywords=OMIT_KEYWORDS,
    )

    mpr_series_uid = pydicom.uid.generate_uid()
    mpr_dcm_dict = {}
    total_instances = 0
    for image in images:
        disc_label = "Unknown"

        if image is not None:
            if image.HasMetaDataKey("label"):
                disc_label = image.GetMetaData("label")
            if disc_label not in mpr_dcm_dict:
                mpr_dcm_dict[disc_label] = []

            # Loop through in reverse order to make sure display order is as
            # expected
            for i in range(image.GetSize()[2] - 1, -1, -1):
                slice_im = image[:, :, i : i + 1]
                slice_arr = np.squeeze(sitk.GetArrayFromImage(slice_im))
                orientation = [
                    float(d)
                    for d in sitk_direction_to_dicom_iop(slice_im.GetDirection())
                ]

                dcm = MRMPRDataset(
                    slice_arr,
                    source_datasets,
                    series_metadata,
                    series_description=series_description,
                    series_instance_uid=mpr_series_uid,
                    position=list(slice_im.GetOrigin()),
                    orientation=orientation,
                    slice_thickness=image.GetSpacing()[-1],
                    pixel_spacing=slice_im.GetSpacing()[0],
                    instance_number=total_instances + 1,
                    series_number=series_number,
                    transfer_syntax_uid=transfer_syntax_uid,
                )
                dcm.ImageComments = disc_label

                mpr_dcm_dict[disc_label].append(dcm)
                total_instances += 1

    return mpr_dcm_dict


class MRMPRDataset(hd.base.SOPClass):
    """Class for creating DICOM datasets for MR Multiplanar Reformats."""

    def __init__(
        self,
        array: np.ndarray,
        ref_datasets: List[pydicom.Dataset],
        series_metadata: pydicom.Dataset,
        instance_number: int,
        position: List[float],
        orientation: List[float],
        pixel_spacing: float,
        slice_thickness: float,
        series_instance_uid: Union[str, pydicom.uid.UID],
        series_number: int,
        series_description: str = "MPR",
        transfer_syntax_uid: str = JPEGLSLossless,
    ):
        """
        Parameters
        ----------
        array: np.ndarray
            2D numpy array representing the pixels for the new MPR/MIP instance
        ref_datasets: List[pydicom.dataset.Dataset]
            List of all datasets from the original CT series used to create the
            MPR/MIP array. These will be used to populate basic metadata and
            set-up the derivation information correctly.
        series_metadata: pydicom.Dataset
            Dataset containing other metadata information that should be placed
            into every image in the new series.  NB this does not need to be a
            valid DICOM dataset.
        instance_number: int
            Instance number for the new DICOM dataset
        position: List[float]
            Value for the ImagePositionPatient tag of the new DICOM dataset
        orientation: List[float]
            Value for the ImageOrientationPatient tag of the new DICOM dataset
        pixel_spacing: float
            Pixel spacing within the imaging plane of 'array'. Assumed to be isotropic
        slice_thickness: float
            Slice thickness value used to create the MPR/MIP (in mm)
        series_instance_uid: Union[str, pydicom.uid.UID]
            Series instance UID for the series to which the new instance should belong
        series_number: int
            Series number for the new series
        series_description: str
            Series description for the series to which the new instance will belong
        transfer_syntax_uid: str
            Transfer syntax uid
        """

        supported_transfer_syntaxes = {
            ImplicitVRLittleEndian,
            ExplicitVRLittleEndian,
            RLELossless,
            JPEGBaseline8Bit,
            JPEG2000Lossless,
            JPEGLSLossless,
        }
        if transfer_syntax_uid not in supported_transfer_syntaxes:
            raise ValueError(
                f'Transfer syntax "{transfer_syntax_uid}" is not supported'
            )

        # the SOP Class UID for MR Image Storage
        mr_sop_class_uid = "1.2.840.10008.5.1.4.1.1.4"
        sop_instance_uid = pydicom.uid.generate_uid()
        ref = ref_datasets[0]
        super().__init__(
            study_instance_uid=ref.StudyInstanceUID,
            series_instance_uid=series_instance_uid,
            series_number=series_number,
            sop_instance_uid=sop_instance_uid,
            sop_class_uid=mr_sop_class_uid,
            instance_number=instance_number,
            modality="MR",
            manufacturer=MANUFACTURER,
            patient_id=getattr(ref, "PatientID", None),
            patient_name=getattr(ref, "PatientName", None),
            patient_birth_date=getattr(ref, "PatientBirthDate", None),
            patient_sex=getattr(ref, "PatientSex", None),
            accession_number=getattr(ref, "AccessionNumber", None),
            study_id=getattr(ref, "StudyID", None),
            study_date=getattr(ref, "StudyDate", None),
            study_time=getattr(ref, "StudyTime", None),
            referring_physician_name=getattr(ref, "ReferringPhysicianName", None),
            series_description=series_description,
            transfer_syntax_uid=transfer_syntax_uid,
        )

        self.ImageType = ["DERIVED", "SECONDARY", "MPR"]
        self.SoftwareVersions = "prostate"  # __version__

        # Content and acquisition times. Per the standard should be set to the
        # earliest of the source images
        if all("AcquisitionTime" in ds for ds in ref_datasets):
            sorted_acq_times = sorted(ref_datasets, key=lambda ds: ds.AcquisitionTime)
            self.AcquisitionTime = sorted_acq_times[0].AcquisitionTime
        if all("ContentTime" in ds for ds in ref_datasets):
            sorted_content_times = sorted(ref_datasets, key=lambda ds: ds.ContentTime)
            self.ContentTime = sorted_content_times[0].ContentTime

        # Instance Creation Time (now)
        self.InstanceCreationTime = datetime.now().time()
        self.InstanceCreationDate = datetime.today()

        # Derivation description specifies the sequence of steps used to derive
        # this image from the source images See
        # http://dicom.nema.org/medical/dicom/current/output/chtml/part16/sect_CID_7203.html
        # for reference of the DCM coding scheme used here

        # The multiplanar reformat step
        reformat_step = pydicom.Dataset()
        reformat_step.CodeValue = "113072"
        reformat_step.CodingSchemeDesignator = "DCM"
        reformat_step.CodeMeaning = "Multiplanar reformatting"

        self.DerivationCodeSequence = pydicom.Sequence([reformat_step])
        self.DerivationDescription = (
            "Angle-corrected multi-planar reformat of the vertebral disc region"
        )

        # Source images information - specifies which instances were used as
        # the source of the information for this images. We will use every
        # slice of the input image here.
        src_images = []
        for ds in ref_datasets:
            src_ins = pydicom.Dataset()
            src_ins.ReferencedSOPClassUID = ds.SOPClassUID
            src_ins.ReferencedSOPInstanceUID = ds.SOPInstanceUID
            src_ins.SpatialLocationsPreserved = "YES"
            src_images.append(src_ins)
        self.SourceImageSequence = pydicom.Sequence(src_images)

        # Copy frame of reference UID
        try:
            self.FrameOfReferenceUID = ref_datasets[0].FrameOfReferenceUID
        except AttributeError:
            # Occasioanlly anonymized studies are missing frame of reference UID
            pass

        # Store the provided geometric information
        self.ImagePositionPatient = [
            pydicom.valuerep.DSfloat(format_number_as_ds(pos)) for pos in position
        ]
        self.ImageOrientationPatient = [
            pydicom.valuerep.DSfloat(format_number_as_ds(ori)) for ori in orientation
        ]
        self.PixelSpacing = [pixel_spacing, pixel_spacing]
        self.SliceThickness = slice_thickness

        # Pixel interpretation information
        self.Rows = array.shape[0]
        self.Columns = array.shape[1]
        self.SamplesPerPixel = 1
        self.PixelRepresentation = 0
        self.BitsAllocated = ref.BitsAllocated  # 16
        self.BitsStored = ref.BitsStored  # 16
        self.HighBit = ref.BitsStored - 1  # 15
        self.PhotometricInterpretation = ref.PhotometricInterpretation

        # Unclear how to handle these if they differ in input slices
        if "RepetitionTime" not in series_metadata:
            self.RepetitionTime = None
        if "EchoTime" not in series_metadata:
            self.EchoTime = None

            #
        # Pixels
        rescale_intercept = array.min()
        self.RescaleIntercept = rescale_intercept
        self.RescaleSlope = 1.0
        if self.BitsAllocated == 8:
            pixel_array = (array - rescale_intercept).astype(np.uint8)  # .tobytes()

        elif self.BitsAllocated == 16:
            pixel_array = (array - rescale_intercept).astype(np.uint16)  # .tobytes()

        encoded_frame = encode_frame(
            pixel_array,
            transfer_syntax_uid=transfer_syntax_uid,
            bits_allocated=self.BitsAllocated,
            bits_stored=self.BitsStored,
            photometric_interpretation=self.PhotometricInterpretation,
            pixel_representation=self.PixelRepresentation,
            planar_configuration=getattr(self, "PlanarConfiguration", None),
        )
        if self.file_meta.TransferSyntaxUID.is_encapsulated:
            self.PixelData = encapsulate([encoded_frame])
        else:
            self.PixelData = encoded_frame

        # self.add(pixel_data)
        self.WindowCenter = ref_datasets[0].WindowCenter
        self.WindowWidth = ref_datasets[0].WindowWidth

        # Copy over remaining metadata from the series dataset
        for el in series_metadata.elements():
            if el.tag not in self:
                self.add(el)
