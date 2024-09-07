"""
`preprocessing` is a python library designed for the purpose of preprocessing MRI
data at QTIM. It currently supports reorganization of DICOM and NIfTI files to follow
a BIDS inspired convention, DICOM to NIfTI conversion, and preprocessing for brain data. Its
outputs are also intended to follow a BIDS inspired organizational scheme.

Public Packages
---------------
data
    The `data` package contains tools for organizing DICOM and NIfTI datasets into
    a BIDS inspired organizational scheme and converting files from DICOM to NIfTI format.

brain
    The `brain` package contains tools specific to preprocessing brain data, such as
    skullstripping and the brain preprocessing pipeline.

qc
    The `qc` package contains tools related to data quality control.

Public Modules
--------------
constants
    The `constants` module contains important constants that are referenced frequently
    throughout the rest of the library.

dcm_tools
    The `dcm_tools` module contains code relevant for analyzing DICOM files.

synthmorph
    The `synthmorph` module uses the Synthmorph models to perform image registration.

utils
    The `utils` module contains custom exceptions and useful functions that are referenced
    frequently throughout the rest of the library.
"""

__all__ = [
    "data",
    "brain",
    "qc",
    "constants",
    "dcm_tools",
    "synthmorph",
    "utils",
]
