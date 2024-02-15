import os
from sys import argv

# check CLI arguments for gpu usage and override to cpu if multiprocessing
# if accessing through python and want to use both, import tensorflow first
use_gpu = "-g" in argv or "--gpu" in argv
use_multiprocessing = "-c" in argv or "--cpu" in argv

os.environ["CUDA_VISIBLE_DEVICES"] = (
    os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if use_gpu and not use_multiprocessing
    else ""
)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.environ.get("TF_CPP_MIN_LOG_LEVEL", "3")


from . import brain
from . import bids


__all__ = [
    "brain",
    "bids",
]
