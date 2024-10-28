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

anonymize_df
    Apply automated anonymization to a DataFrame. This function assumes
    that the 'PatientID' and 'StudyID' tags are consistent and correct
    to derive AnonPatientID = 'sub_{i:02d}' and AnonStudyID = 'ses_{i:02d}'.

create_dicom_dataset
    Create a DICOM dataset CSV compatible with subsequent `preprocessing`
    scripts. The final CSV provides a series level summary of the location
    of each series alongside metadata extracted from DICOM headers.  If the
    previous organization schems of the dataset does not enforce a DICOM
    series being isolated to a unique directory (instances belonging to
    multiple series must not share the same lowest level directory),
    reorganization must be applied for NIfTI conversion.

create_nifti_dataset
    Create a NIfTI dataset CSV compatible with subsequent `preprocessing`
    scripts. The final CSV provides a series level summary of the location
    of each series alongside metadata generated to simulate DICOM headers.
    Specifically, ['PatientID', 'StudyDate', 'SeriesInstanceUID',
    'SeriesDescription', 'StudyInstanceUID'] (and optionally
    'NormalizedSeriesDescription') are inferred or randomly generated.
"""

from .nifti_conversion import convert_to_nifti, convert_study, convert_batch_to_nifti
from .datasets import anonymize_df, create_nifti_dataset, create_dicom_dataset


__all__ = [
    "convert_to_nifti",
    "convert_study",
    "convert_batch_to_nifti",
    "anonymize_df",
    "create_nifti_dataset",
    "create_dicom_dataset",
]
