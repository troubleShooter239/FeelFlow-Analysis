from os import getenv
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
from requests import get
from PIL import Image
from base64 import b64decode
from numba import njit
from keras.preprocessing.image import img_to_array

from opencv_client import DetectorWrapper


def get_deepface_home() -> str:
    """Get the home directory for storing weights and models.

    Returns:
        str: the home directory."""
    return str(getenv("DEEPFACE_HOME", default=str(Path.home())))


@njit
def l2_normalize(x: Union[np.ndarray, list]) -> np.ndarray:
    if isinstance(x, list):
        x = np.array(x)
    return x / np.sqrt(np.sum(np.multiply(x, x)))


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


def extract_faces(img: Union[str, np.ndarray], target_size: tuple = (224, 224), 
                  grayscale: bool = False, enforce_detection: bool = True, 
                  align: bool = True) -> List[Tuple[np.ndarray, Dict[str, int], float]]:
    """Extract faces from an image.
    Args:
        img: a path, url, base64 or numpy array.
        target_size (tuple, optional): the target size of the extracted faces.
        Defaults to (224, 224).
        detector_backend (str, optional): the face detector backend. Defaults to "opencv".
        grayscale (bool, optional): whether to convert the extracted faces to grayscale.
        Defaults to False.
        enforce_detection (bool, optional): whether to enforce face detection. Defaults to True.
        align (bool, optional): whether to align the extracted faces. Defaults to True.

    Raises:
        ValueError: if face could not be detected and enforce_detection is True.

    Returns:
        results (List[Tuple[np.ndarray, dict, float]]): A list of tuples
            where each tuple contains:
            - detected_face (np.ndarray): The detected face as a NumPy array.
            - face_region (dict): The image region represented as
                {"x": x, "y": y, "w": w, "h": h}
            - confidence (float): The confidence score associated with the detected face."""
    img, img_name = load_image(img)

    face_objs = DetectorWrapper.detect_faces(img, align)

    if len(face_objs) == 0 and enforce_detection:
        if img_name is not None:
            raise ValueError(f"Face could not be detected in {img_name}."
                             "Please confirm that the picture is a face photo "
                             "or consider to set enforce_detection param to False.")
        else:
            raise ValueError("Face could not be detected. Please confirm that the "
                             "picture is a face photo or consider to set "
                             "enforce_detection param to False.")

    img_region = [0, 0, img.shape[1], img.shape[0]]
    if len(face_objs) == 0 and not enforce_detection:
        face_objs = [(img, img_region, 0)]

    extracted_faces = []
    for current_img, reg, confidence in face_objs:
        if current_img.shape[0] < 0 or current_img.shape[1] < 0:
            continue

        if grayscale:
            current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

        factor = min(target_size[0] / current_img.shape[0], 
                     target_size[1] / current_img.shape[1])

        current_img = cv2.resize(current_img, 
                                (int(current_img.shape[1] * factor), 
                                int(current_img.shape[0] * factor)))

        diff_0 = target_size[0] - current_img.shape[0]
        diff_1 = target_size[1] - current_img.shape[1]
        if not grayscale:
            current_img = np.pad(current_img, 
                                ((diff_0 // 2, diff_0 - diff_0 // 2),
                                (diff_1 // 2, diff_1 - diff_1 // 2),
                                (0, 0)), "constant")
        else:
            current_img = np.pad(current_img, 
                                ((diff_0 // 2, diff_0 - diff_0 // 2),
                                (diff_1 // 2, diff_1 - diff_1 // 2)), "constant")

        if current_img.shape[0:2] != target_size:
            current_img = cv2.resize(current_img, target_size)

        img_pixels = np.expand_dims(img_to_array(current_img), axis=0) / 255
        regs = {"x": int(reg[0]), "y": int(reg[1]), "w": int(reg[2]), "h": int(reg[3])}
        extracted_faces.append((img_pixels, regs, confidence))

    if len(extracted_faces) == 0 and enforce_detection:
        raise ValueError(f"Detected face shape is {img.shape}. "
                         "Consider to set enforce_detection arg to False.")

    return extracted_faces


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
    return sizes[model_name]