import os
from pathlib import Path
from bids_validator import BIDSValidator
from typing import List


def validate(rootpath: Path | str, paths: Path | str | List[Path | str]):
    """
    Validate path(s) are compliant with BIDS.

    Parameters
    __________
    rootpath: Path | str
        The path to the root of the BIDS dataset.
    paths: Path | str | List[Path | str]
        The path to file(s) to check against the BIDSValidator. They must be relative to
        'rootpath', but include a leading '/'.

    Returns
    _______
    bool | List[bool]:
        A bool or list of bools indicating whether a file is compliant with BIDS.
    """
    os.chdir(rootpath)

    validator = BIDSValidator()

    if isinstance(paths, List):
        is_bids = []
        for path in paths:
            is_bids.append(validator.is_bids(path))
        return is_bids
    else:
        return validator.is_bids(paths)
