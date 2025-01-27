"""
The `nifti_conversion` module contains tools for converting from DICOM to NIfTI format.

Public Functions
----------------
convert_series
    Convert a DICOM series to a NIfTI file.

convert_study
    Convert a DICOM study to NIfTI files representing each series.

convert_batch_to_nifti
    Convert a DICOM dataset to NIfTI files representing each series.
"""
from warnings import warn
import pandas as pd
import numpy as np
import datetime
import highdicom as hd
import SimpleITK as sitk

from typing import Literal
from tqdm import tqdm
from pathlib import Path
from preprocessing.utils import (
    check_required_columns,
    hd_to_sitk,
    update_errorfile
)
from preprocessing.dcm_tools import sort_slices, calc_slice_distance
from pydicom import dcmread
from numpy.linalg import norm
from concurrent.futures import ProcessPoolExecutor, as_completed


def dicom_integrity_checks(series_dir: Path | str, eps: float = 1e-3) -> bool:
    """
    Check the integrity of a DICOM series. This includes verification of consistent
    values for the 'ImageOrientationPatient', 'SliceThickness', and 'PixelSpacing'
    tags. 'ImageOrientationPatient' must also form an orthonormal basis.

    Parameters
    ----------
    series_dir: Path | str
        The path to a directory containing a DICOM series. All of the DICOM instances
        are assumed to have the .dcm file extension.

    eps: float
        The absolute error tolerance allowed to pass the integrity checks. Defaults to 1e-3.

    Returns
    -------
    bool
        The success or failure of the integrity checks are represented as True or False
        respectively.
    """
    series_dir = Path(series_dir).resolve()

    files = list(series_dir.glob("**/*.dcm"))

    dcms = []

    for file in files:
        dcms.append(dcmread(file, stop_before_pixels=True))

    dcms = sort_slices(dcms)

    # required metadata is present and consistent
    required_metadata = ["ImageOrientationPatient", "SliceThickness", "PixelSpacing"]

    try:
        for meta in required_metadata:
            meta_0 = getattr(dcms[0], meta)
            for dcm in dcms[1:]:
                if not np.allclose(meta_0, getattr(dcm, meta), atol=eps):
                    return False
    except Exception:
        return False

    # orientaation is an orthonormal basis
    orientation = np.array(dcms[0].ImageOrientationPatient)

    if not np.allclose(norm(orientation[:3]), 1, atol=eps) or not np.allclose(
        norm(orientation[3:]), 1, atol=eps
    ):
        return False

    if not np.isclose(orientation[:3].dot(orientation[3:]), 0, atol=eps):
        return False

    # consistent distance
    dists = []
    for dcm in dcms:
        dists.append(
            calc_slice_distance(dcm.ImageOrientationPatient, dcm.ImagePositionPatient)
        )

    spacing = np.diff(dists)

    if not np.allclose(spacing, spacing.mean(), atol=eps):
        return False

    return True


def convert_series(
    dicom_dir: Path | str,
    nifti_dir: Path | str,
    anon_patient_id: str,
    anon_study_id: str,
    normalized_series_description: str,
    subdir: Literal["anat", "func", "dwi"] = "anat",
    overwrite: bool = False,
) -> str | None:
    """
    Convert a DICOM series to a NIfTI file.

    Parameters
    ----------
    dicom_dir: Path | str
        The path to a directory containing all of the DICOM instances in a single series.

    nifti_dir: Path | str
        The root directory under which the converted NIfTI files will be written. Subdirectories
        will be created to follow a BIDS inspired convention.

    anon_patient_id: str
        The anonymized PatientID for the series being converted (e.g., 'sub-01').

    anon_study_id: str
        The anonymized StudyID for the series being converted (e.g., 'ses-01').

    manufacturer: str
        The manufacturer information originally stored in the DICOM header.

    normalized_series_description: str
        The series_description normalized to a consistent value within a dataset.

    subdir: Literal['anat', 'func', 'dwi']
        The subdirectory under the study directory. This represents the modality information for
        BIDS. Currently, the supported options are 'anat', 'func', and 'dwi'.

    overwrite: bool
        Whether to overwrite the NIfTI file if there is already one with the same output name.
        Defaults to False.

    Returns
    -------
    str | None
        The output name of the NIfTI file if it is successfully created, else None.
    """
    image_converted = False

    nifti_dir = Path(nifti_dir).resolve()

    output_dir = nifti_dir / anon_patient_id / anon_study_id / subdir
    output_nifti = (
        output_dir
        / f"{anon_patient_id}_{anon_study_id}_{normalized_series_description.replace(' ', '_')}.nii.gz"
    )

    if Path(output_nifti).exists() and not overwrite:
        return str(output_nifti)


    if not dicom_integrity_checks(dicom_dir):
        print(
            f"{dicom_dir} does not pass integrity checks and will not be converted to NIfTI"
        )
        return ''

    output_dir.mkdir(parents=True, exist_ok=True)

    dicom_dir = Path(dicom_dir).resolve()

    files = list(dicom_dir.glob("**/*.dcm"))

    dcms = []

    for file in files:
        dcms.append(dcmread(file, stop_before_pixels=False))

    dcm_groups = sort_slices(dcms, group_by_position=True)

    for i, group in enumerate(dcm_groups):
        try:
            hd_im = hd.image.get_volume_from_series(group, atol=1e-3)

        except Exception as error:
            #print([hasattr(d, "PixelData") for d in group])
            print(error)
            continue

        sitk_im = hd_to_sitk(hd_im)

        outfile = str(output_nifti) if i == 0 else str(output_nifti).replace(".nii.gz", f"_{i}.nii.gz")
        sitk.WriteImage(sitk_im, outfile)

        image_converted = True

    if image_converted:
        return str(output_nifti)

    return ''



def convert_seg(
    dicom_dir: Path | str,
    output_nifti: Path | str,
    overwrite: bool = False,
) -> str | None:
    if Path(output_nifti).exists() and not overwrite:
        return str(output_nifti)

    segfile = list(Path(dicom_dir).glob("*"))[0]

    seg = hd.seg.segread(segfile)
    vol_hd = seg.get_volume(combine_segments=True)

    # Work around sitk's infuriating transposition
    vol_arr = np.transpose(vol_hd.array, [2, 1, 0])

    vol_sitk = sitk.GetImageFromArray(vol_arr)
    vol_sitk.SetOrigin(vol_hd.position)
    vol_sitk.SetSpacing(vol_hd.spacing)
    vol_sitk.SetDirection(vol_hd.direction.flatten())

    sitk.WriteImage(vol_sitk, output_nifti)

    return str(output_nifti)


def convert_study(
    study_df: pd.DataFrame,
    nifti_dir: Path | str,
    seg_source: str = "T1Post",
    overwrite_nifti: bool = False,
    check_columns: bool = True,
) -> pd.DataFrame:
    """
    Convert a DICOM study to NIfTI files representing each series.

    Parameters
    ----------
    study_df: pd.DataFrame
        A DataFrame containing data for a single study. It must contain the following
        columns: ['Dicoms', 'AnonPatientID', 'AnonStudyID', 'StudyInstanceUID',
        'Manufacturer', 'NormalizedSeriesDescription', 'SeriesType'].

    nifti_dir: Path | str
        The root directory under which the converted NIfTI files will be written. Subdirectories
        will be created to follow a BIDS inspired convention.

    overwrite: bool
        Whether to overwrite the NIfTI file if there is already one with the same output name.
        Defaults to False.

    check_columns: bool
        Whether to check 'study_df' for the required columns. Defaults to True.

    Returns
    -------
    pd.DataFrame
        A DataFrame that contains the new column: 'Nifti'
    """
    if check_columns:
        required_columns = [
            "Dicoms",
            "AnonPatientID",
            "AnonStudyID",
            "StudyInstanceUID",
            "Manufacturer",
            "SeriesDescription",
            "SeriesType",
            "Modality"
        ]

        check_required_columns(study_df, required_columns)

    rows = study_df.to_dict("records")

    encountered_series_descriptions = []

    for row in rows: # i, row in enumerate(rows):
        if row["Modality"] != "SEG":
            if pd.isna(row["NormalizedSeriesDescription"]) or row["NormalizedSeriesDescription"] in encountered_series_descriptions:
                continue

            row["Nifti"] = convert_series(
                dicom_dir=row["Dicoms"],
                nifti_dir=nifti_dir,
                anon_patient_id=row["AnonPatientID"],
                anon_study_id=row["AnonStudyID"],
                normalized_series_description=row["NormalizedSeriesDescription"],
                subdir=row["SeriesType"],
                overwrite=overwrite_nifti,
            )

        else:
            for r in rows:
                if r["NormalizedSeriesDescription"] == seg_source and Path(r["Nifti"]).exists():

                    output_nifti = r["Nifti"].replace(".nii.gz", "_seg.nii.gz")

                    convert_seg(
                        dicom_dir=row["Dicoms"],
                        output_nifti=output_nifti,
                        overwrite=overwrite_nifti
                    )

                    r["Seg"] = output_nifti

                    break

    return pd.DataFrame(rows)


def convert_batch_to_nifti(
    nifti_dir: Path | str,
    csv: Path | str,
    overwrite_nifti: bool = False,
    cpus: int = 1,
    check_columns: bool = True,
) -> pd.DataFrame:
    """
    Convert a DICOM dataset to NIfTI files representing each series.

    Parameters
    ----------
    nifti_dir: Path | str
        The root directory under which the converted NIfTI files will be written. Subdirectories
        will be created to follow a BIDS inspired convention.

    csv: Path | str
        The path to a CSV containing an entire dataset. It must contain the following
        columns: ['Dicoms', 'AnonPatientID', 'AnonStudyID', 'StudyInstanceUID',
        'SeriesInstanceUID', 'Manufacturer', 'NormalizedSeriesDescription', 'SeriesType'].

    overwrite: bool
        Whether to overwrite the NIfTI file if there is already one with the same output name.
        Defaults to False.

    cpus: int
        Number of cpus to use for multiprocessing. Defaults to 1 (no multiprocessing).

    check_columns: bool
        Whether to check the CSV for the required columns. Defaults to True.

    Returns
    -------
    pd.DataFrame
        A DataFrame that contains the new column: 'Nifti'. This DataFrame will be used to overwrite
        the CSV.
    """
    df = pd.read_csv(csv, dtype=str)

    if check_columns:
        required_columns = [
            "Dicoms",
            "AnonPatientID",
            "AnonStudyID",
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "Manufacturer",
            "NormalizedSeriesDescription",
            "SeriesType",
            "Modality"
        ]

        check_required_columns(df, required_columns)

    if "Nifti" not in df.columns:
        df["Nifti"] = [""] * df.shape[0]

    filtered_df = df[(~pd.isna(df["NormalizedSeriesDescription"])) | (df["Modality"] == "SEG")]

    study_uids = filtered_df["StudyInstanceUID"].unique()

    nifti_dir = Path(nifti_dir).resolve()
    errorfile = nifti_dir / f"{str(datetime.datetime.now()).replace(' ', '_')}.txt"

    kwargs_list = [
        {
            "study_df": filtered_df[
                filtered_df["StudyInstanceUID"] == study_uid
            ].copy(),
            "nifti_dir": nifti_dir,
            "overwrite_nifti": overwrite_nifti,
            "check_columns": False,
        }
        for study_uid in study_uids
    ]

    with tqdm(
        total=len(kwargs_list), desc="Converting to NIfTI"
    ) as pbar, ProcessPoolExecutor(cpus if cpus >= 1 else 1) as executor:
        futures = {
            executor.submit(convert_study, **kwargs): kwargs
            for kwargs in kwargs_list
        }

        for future in as_completed(futures.keys()):
            try:
                nifti_df = future.result()


                df = (
                    pd.read_csv(csv, dtype=str)
                    .drop_duplicates(subset="SeriesInstanceUID")
                    .reset_index(drop=True)
                )

                # print(f"nifti_df: {nifti_df['NormalizedSeriesDescription'].dtype}, df: {df['NormalizedSeriesDescription'].dtype}")

                df = pd.merge(df, nifti_df, how="outer")  #df.merge(nifti_df, how="left")
                df = (
                    df.drop_duplicates(subset="SeriesInstanceUID")
                    .sort_values(["AnonPatientID", "AnonStudyID"])
                    .reset_index(drop=True)
                )
                df.to_csv(csv, index=False)
                pbar.update(1)

            except Exception as error:
                update_errorfile(
                    func_name="preprocessing.data.convert_study",
                    kwargs=futures[future],
                    errorfile=errorfile,
                    error=error
                )
                print(error)

                pbar.update(1)
                continue


    df = (
        pd.read_csv(csv, dtype=str)
        .drop_duplicates(subset="SeriesInstanceUID")
        .sort_values(["AnonPatientID", "AnonStudyID"])
        .reset_index(drop=True)
    )
    df.to_csv(csv, index=False)
    return df


__all__ = ["convert_series", "convert_seg", "convert_study", "convert_batch_to_nifti"]
