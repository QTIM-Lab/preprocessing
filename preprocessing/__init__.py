"""
`preprocessing` is a python library designed for the purpose of preprocessing MRI
data at QTIM. It currently supports reorganization of DICOM and NIfTI files to follow
BIDS conventions, DICOM to NIfTI conversion, and preprocessing for brain data. Its
outputs are intended to follow the BIDS organizational scheme.

Public Packages
_______________
bids
    The `bids` package contains tools for organizing DICOM and NIfTI datasets into
    the BIDS organizational scheme and converting files from DICOM to NIfTI format.

brain
    The `brain` package contains tools specific to preprocessing brain data, such as
    skullstripping and the brain preprocessing pipeline.

Public Modules
______________
constants
    The `constants` module contains important constants that are referenced frequently
    throughout the rest of the library.

dcm_tools
    The `dcm_tools` module contains code relevant for analyzing DICOM files.

series_selection
    The `series_selection` module utilizes `mr_series_selection` to predict normalized
    series descriptions for series within a `preprocessing` library compatible
    dataset.

synthmorph
    The `synthmorph` module uses the Synthmorph models to perform image registration.

utils
    The `utils` module contains custom exceptions and useful functions that are referenced
    frequently throughout the rest of the library.
"""

__all__ = [
    "bids",
    "brain",
    "constants",
    "dcm_tools",
    "longitudinal_tracking",
    "series_selection",
    "synthmorph",
    "utils",
]
