from pathlib import Path
from bids_validator import BIDSValidator
from typing import Sequence


def validate(paths: Path | str | Sequence[Path | str]):
    validator = BIDSValidator()

    if isinstance(paths, Sequence):
        for path in paths:
            validator.is_bids(path)
    else:
        validator.is_bids(paths)
