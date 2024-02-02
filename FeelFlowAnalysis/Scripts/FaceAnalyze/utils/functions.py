import os
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from numba import njit

import utils.constants as C


def get_deepface_home() -> str:
    """Get the home directory for storing weights and models.

    Returns:
        str: the home directory."""
    return str(os.getenv(C._DEEPFACE_HOME, str(Path.home())))


def initialize_folder() -> None:
    """Initialize the folder for storing weights and models.

    Raises:
        OSError: if the folder cannot be created."""
    deepFaceHomePath = get_deepface_home() + "/.deepface"
    if not os.path.exists(deepFaceHomePath):
        os.makedirs(deepFaceHomePath, exist_ok=True)
    weightsPath = deepFaceHomePath + "/weights"
    if not os.path.exists(weightsPath):
        os.makedirs(weightsPath, exist_ok=True)


@njit
def l2_normalize(x: Union[np.ndarray, list]) -> np.ndarray:
    if isinstance(x, list):
        x = np.array(x)
    return x / np.sqrt(np.sum(np.multiply(x, x)))


def find_threshold(model_name: str, distance_metric: str) -> float:
    base_threshold = {"cosine": 0.40, "euclidean": 0.55, "euclidean_l2": 0.75}
    thresholds = {
        "VGG-Face": {"cosine": 0.68, "euclidean": 1.17, "euclidean_l2": 1.17},
        "Facenet": {"cosine": 0.40, "euclidean": 10, "euclidean_l2": 0.80},
        "Facenet512": {"cosine": 0.30, "euclidean": 23.56, "euclidean_l2": 1.04},
        "ArcFace": {"cosine": 0.68, "euclidean": 4.15, "euclidean_l2": 1.13},
        "Dlib": {"cosine": 0.07, "euclidean": 0.6, "euclidean_l2": 0.4},
        "SFace": {"cosine": 0.593, "euclidean": 10.734, "euclidean_l2": 1.055},
        "OpenFace": {"cosine": 0.10, "euclidean": 0.55, "euclidean_l2": 0.55},
        "DeepFace": {"cosine": 0.23, "euclidean": 64, "euclidean_l2": 0.64},
        "DeepID": {"cosine": 0.015, "euclidean": 45, "euclidean_l2": 0.17}
    }
    return thresholds.get(model_name, base_threshold).get(distance_metric, 0.4)


@njit
def normalize_input(img: np.ndarray, normalization: str = "base") -> np.ndarray:
    """Normalize input image.

    Args:
        img (numpy array): the input image.
        normalization (str, optional): the normalization technique. Defaults to "base",
        for no normalization.

    Returns:
        numpy array: the normalized image."""
    if normalization == "base":
        return img
    img *= 255
    if normalization == "raw":
        pass
    elif normalization == "Facenet":
        mean, std = img.mean(), img.std()
        img = (img - mean) / std
    elif normalization == "Facenet2018":
        img = img / 127.5 - 1
    elif normalization == "VGGFace":
        img[..., 0] -= 93.5940
        img[..., 1] -= 104.7624
        img[..., 2] -= 129.1863
    elif normalization == "VGGFace2":
        img[..., 0] -= 91.4953
        img[..., 1] -= 103.8827
        img[..., 2] -= 131.0912
    elif normalization == "ArcFace":
        img = (img - 127.5) / 128
    else:
        raise ValueError(f"unimplemented normalization type - {normalization}")
    return img


def find_size(model_name: str) -> Tuple[int, int]:
    """Find the target size of the model.

    Args:
        model_name (str): the model name.

    Returns:
        tuple: the target size."""
    sizes = {
        "VGG-Face": (224, 224),
        "Facenet": (160, 160),
        "Facenet512": (160, 160),
        "OpenFace": (96, 96),
        "DeepFace": (152, 152),
        "DeepID": (47, 55),
        "Dlib": (150, 150),
        "ArcFace": (112, 112),
        "SFace": (112, 112),
    }
    try:
        return sizes[model_name]
    except KeyError:
        return (0, 0)
