"""
The `data` package contains tools for organizing DICOM and NIfTI datasets into a
BIDS inspired organizational scheme and converting files from DICOM to NIfTI format.

Public Functions
----------------
convert_to_nifti
    Convert a DICOM series to a NIfTI file.

convert_study
    Convert a DICOM study to NIfTI files representing each series.

convert_batch_to_nifti
    Convert a DICOM dataset to NIfTI files representing each series.

find_anon_keys
    Create anonymization keys for anonymous PatientID and StudyID from previous
    QTIM organizational scheme. Is compatible with data following a following
    <Patient_ID>/<Study_ID> directory hierarchy.

nifti_anon_csv
    Create anonymization keys for a dataset that starts within NIfTI format. If the
    'SeriesDescription's are not normalized, 'NormalizedSeriesDescription's must be
    obtained externally before the NIfTI dataset can be reorganized.

reorganize_dicoms
    Reorganize DICOMs to follow a BIDS inspired convention. Any DICOMs found recursively
    within this directory will be reorganized (at least one level of subdirectories
    is assumed). Anonomyzation keys for PatientIDs and StudyIDs are provided within
    a CSV.

reorganize_niftis
    Reorganize a NIfTI dataset to follow a BIDS inspired convention. As NIfTI files lack metadata,
    anonymization keys must be provided in the form of a CSV, such as one obtained with
    `nifti_anon_csv`.
"""

from .reorganize import (
    find_anon_keys,
    reorganize_dicoms,
    nifti_anon_csv,
    reorganize_niftis,
)
from .nifti_conversion import convert_to_nifti, convert_study, convert_batch_to_nifti
from .anonymization import anonymize_df
from .datasets import create_nifti_dataset


__all__ = [
    "convert_to_nifti",
    "convert_study",
    "convert_batch_to_nifti",
    "find_anon_keys",
    "nifti_anon_csv",
    "reorganize_dicoms",
    "reorganize_niftis",
    "anonymize_df",
    "create_nifti_dataset"
]
