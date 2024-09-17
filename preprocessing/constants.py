"""
The `constants` module contains important constants that are referenced frequently
throughout the rest of the library.

Public Constants
----------------
META_KEYS
    A list of useful metadata that should be extracted from the DICOM header if
    available.

PREPROCESSING_MODELS_PATH
    The directory in which the Synthmorph and Synthstrip models are stored.
"""

import os
from .utils import initialize_models

META_KEYS = [
    "SeriesInstanceUID",
    "StudyInstanceUID",
    "PatientID",
    "AccessionNumber",
    "Manufacturer",
    "StudyDate",
    "StudyDescription",
    "SeriesDescription",
    "Modality",
]

REQUIRED_KEYS = [
    "SeriesInstanceUID",
    "StudyInstanceUID",
    "PatientID",
    "StudyDate",
]


PREPROCESSING_MODELS_PATH = (
    os.environ["PREPROCESSING_MODELS_PATH"]
    if "PREPROCESSING_MODELS_PATH" in os.environ
    else initialize_models()
)

__all__ = ["META_KEYS", "REQUIRED_KEYS", "PREPROCESSING_MODELS_PATH"]
