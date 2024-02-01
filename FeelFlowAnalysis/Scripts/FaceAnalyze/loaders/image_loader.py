from typing import Tuple, Union

import cv2
import numpy as np
from base64 import b64decode
from pathlib import Path
from PIL import Image
from requests import get


def load_base64(uri: str) -> np.ndarray:
    """Load image from base64 string.
    Args:
        uri: a base64 string.
    Returns:
        numpy array: the loaded image."""
    return cv2.imdecode(np.fromstring(b64decode(uri.split(",")[1]), np.uint8), 
                        cv2.IMREAD_COLOR)


def load_image(img: Union[str, np.ndarray]) -> Tuple[np.ndarray, str]:
    """Load image from path, url, base64 or numpy array.
    Args:
        img: a path, url, base64 or numpy array.
    Returns:
        image (numpy array): the loaded image in BGR format
        image name (str): image name itself"""
    if isinstance(img, np.ndarray):
        return img, "numpy array"
    if isinstance(img, Path):
        img = str(img)
    if img.startswith("data:image/"):
        return load_base64(img), "base64 encoded string"
    if img.startswith("http"):
        return (np.array(Image.open(get(img, stream=True, 
                                        timeout=60).raw).convert("BGR")), img)
    return cv2.imread(img), img
