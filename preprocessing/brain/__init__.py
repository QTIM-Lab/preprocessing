"""
The `brain` package contains tools specific to preprocessing brain data, such as
skullstripping and the brain preprocessing pipeline.

Public Functions
________________
preprocess_study
    Preprocess a single study from a DataFrame.

preprocess_patient
    Preprocess all of the studies for a patient in a DataFrame.

preprocess_from_csv
    Preprocess all of the studies within a dataset.

synthstrip_skullstrip
    Adaptation of Freesurfer's mri_synthstrip command. One of `out`, `m`, or `d` must
    be specified.
"""

from .brain_preprocessing import (
    preprocess_study,
    preprocess_patient,
    preprocess_from_csv,
)
from .synthstrip import synthstrip_skullstrip

__all__ = [
    "preprocess_study",
    "preprocess_patient",
    "preprocess_from_csv",
    "synthstrip_skullstrip",
]
