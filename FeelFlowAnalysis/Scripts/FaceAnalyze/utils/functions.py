from os import getenv
from pathlib import Path
from typing import Union

from numba import njit
from numpy import ndarray, array, sqrt, sum, multiply

def get_deepface_home() -> str:
    """Get the home directory for storing weights and models.

    Returns:
        str: the home directory."""
    return str(getenv("DEEPFACE_HOME", default=str(Path.home())))


@njit
def l2_normalize(x: Union[ndarray, list]) -> ndarray:
    if isinstance(x, list):
        x = array(x)
    return x / sqrt(sum(multiply(x, x)))
