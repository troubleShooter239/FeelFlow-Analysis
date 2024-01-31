from os import getenv
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from base_models import FacialRecognitionBase
from base64 import b64decode
from requests import get
from PIL import Image
from numba import njit
from tensorflow.keras.preprocessing.image import img_to_array

from modeling import build_model
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


@njit
def find_cosine_distance(source: Union[np.ndarray, list], test: Union[np.ndarray, list]) -> np.float64:
    if isinstance(source, list):
        source = np.array(source)
    if isinstance(test, list):
        test = np.array(test)
    return 1 - (np.matmul(np.transpose(source), test) / (np.sqrt(np.sum(np.multiply(source, source))) * np.sqrt(np.sum(np.multiply(test, test)))))


@njit
def find_euclidean_distance(source: Union[np.ndarray, list], test: Union[np.ndarray, list]) -> np.float64:
    if isinstance(source, list):
        source = np.array(source)
    if isinstance(test, list):
        test = np.array(test)
    euclidean_distance = source - test
    return np.sqrt(np.sum(np.multiply(euclidean_distance, euclidean_distance)))


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


def represent(img_path: Union[str, np.ndarray], model_name: str = "VGG-Face", 
              enforce_detection: bool = True, detector_backend: str = "opencv",
              align: bool = True, normalization: str = "base") -> List[Dict[str, Any]]:
    """
    Represent facial images as multi-dimensional vector embeddings.

    Args:
        img_path (str or np.ndarray): The exact path to the image, a numpy array in BGR format,
            or a base64 encoded image. If the source image contains multiple faces, the result will
            include information for each detected face.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace and SFace

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Default is True. Set to False to avoid the exception for low-resolution images.

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8'.

        align (boolean): Perform alignment based on the eye positions.

        normalization (string): Normalize the input image before feeding it to the model.
            Default is base. Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace

    Returns:
        results (List[Dict[str, Any]]): A list of dictionaries, each containing the
            following fields:

        - embedding (np.array): Multidimensional vector representing facial features.
            The number of dimensions varies based on the reference model
            (e.g., FaceNet returns 128 dimensions, VGG-Face returns 4096 dimensions).
        - facial_area (dict): Detected facial area by face detection in dictionary format.
            Contains 'x' and 'y' as the left-corner point, and 'w' and 'h'
            as the width and height. If `detector_backend` is set to 'skip', it represents
            the full image area and is nonsensical.
        - face_confidence (float): Confidence score of face detection. If `detector_backend` is set
            to 'skip', the confidence will be 0 and is nonsensical.
    """
    model: FacialRecognitionBase = build_model(model_name)
    target_size = find_size(model_name)
    if detector_backend != "skip":
        img_objs = extract_faces(img_path, (target_size[1], target_size[0]), 
                                 detector_backend, False, enforce_detection, align=align)
    else:
        img, _ = load_image(img_path)
        if len(img.shape) == 4:
            img = img[0]
        if len(img.shape) == 3:
            img = cv2.resize(img, target_size)
            img = np.expand_dims(img, axis=0)
            if img.max() > 1:
                img = (img.astype(np.float32) / 255.0).astype(np.float32)

        img_objs = [(img, {"x": 0, "y": 0, "w": img.shape[1], "h": img.shape[2]}, 0)]
    
    resp_objs = []
    for i, r, c in img_objs:
        e = model.find_embeddings(normalize_input(i, normalization))
        resp_obj = {"embedding": e, "facial_area": r, "face_confidence": c}
        resp_objs.append(resp_obj)

    return resp_objs


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
        
        if grayscale:
            current_img = np.pad(current_img, ((diff_0 // 2, diff_0 - diff_0 // 2),
                                (diff_1 // 2, diff_1 - diff_1 // 2)), "constant")
        else:
            current_img = np.pad(current_img, ((diff_0 // 2, diff_0 - diff_0 // 2),
                                (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), "constant")

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