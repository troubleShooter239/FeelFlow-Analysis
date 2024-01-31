from os import path
from os import stat
from time import ctime, time
from typing import Any, Dict, Union
from json import dumps, loads

from PIL import Image
from PIL.ExifTags import TAGS
import numpy as np
from tensorflow.keras.models import Model

import functions as F
from modeling import build_model
from face_attributes import EmotionClient, GenderClient, RaceClient


def process_age(predictions) -> Dict[str, int]:
    return {"age": int(predictions)}


def process_emotion(predictions) -> Dict[str, Union[Dict[str, float], str]]:
    _sum = predictions.sum()
    return {
        "emotion": {l: round(100 * p / _sum, 2) for l, p in zip(EmotionClient.labels, predictions)},
        "dominant_emotion": EmotionClient.labels[np.argmax(predictions)]
    }


def process_gender(predictions) -> Dict[str, Union[Dict[str, float], str]]:
    return {
        "gender": {l: round(100 * p, 2) for l, p in zip(GenderClient.labels, predictions)},
        "dominant_gender": GenderClient.labels[np.argmax(predictions)]
    }         


def process_race(predictions) -> Dict[str, Union[Dict[str, float], str]]:
    _sum = predictions.sum()
    return {
        "race": {l: round(100 * p / _sum, 2) for l, p in zip(RaceClient.labels, predictions)},
        "dominant_race": RaceClient.labels[np.argmax(predictions)]
    }


def analyze(img: Union[str, np.ndarray], 
            actions: str = '{"age": true, "emotion": true, "gender": true, "race": true}',
            enforce_detection: bool = True, align: bool = True) -> str:
    """
    This function analyzes facial attributes including age, gender, emotion and race.
    In the background, analysis function builds convolutional neural network models to
    classify age, gender, emotion and race of the input image.

    Parameters:
            img: exact image path, numpy array (BGR) or base64 encoded image could be passed.
            If source image has more than one face, then result will be size of number of faces
            appearing in the image.

            actions (tuple): The default is ('age', 'gender', 'emotion', 'race'). You can drop
            some of those attributes.

            enforce_detection (bool): The function throws exception if no face detected by default.
            Set this to False if you don't want to get exception. This might be convenient for low
            resolution images.

            detector_backend (string): set face detector backend to opencv, retinaface, mtcnn, ssd,
            dlib, mediapipe or yolov8.

            align (boolean): alignment according to the eye positions.

            silent (boolean): disable (some) log messages

    Returns:
            The function returns a string of dictionaries for each face appearing in the image.

            [
                    {
                            "region": {'x': 230, 'y': 120, 'w': 36, 'h': 45},
                            "age": 28.66,
                            'face_confidence': 0.9993908405303955,
                            "dominant_gender": "Woman",
                            "gender": {
                                    'Woman': 99.99407529830933,
                                    'Man': 0.005928758764639497,
                            }
                            "dominant_emotion": "neutral",
                            "emotion": {
                                    'sad': 37.65260875225067,
                                    'angry': 0.15512987738475204,
                                    'surprise': 0.0022171278033056296,
                                    'fear': 1.2489334680140018,
                                    'happy': 4.609785228967667,
                                    'disgust': 9.698561953541684e-07,
                                    'neutral': 56.33133053779602
                            }
                            "dominant_race": "white",
                            "race": {
                                    'indian': 0.5480832420289516,
                                    'asian': 0.7830780930817127,
                                    'latino hispanic': 2.0677512511610985,
                                    'black': 0.06337375962175429,
                                    'middle eastern': 3.088453598320484,
                                    'white': 93.44925880432129
                            }
                    }
            ]
    """
    funcs = {"age": process_age, "emotion": process_emotion, "gender": process_gender, "race": process_race}
    
    img_objs = F.extract_faces(img, (224, 224), False, enforce_detection, align)

    models: Dict[str, Model] = {a: build_model(a.capitalize()) for a, s in loads(actions).items() if s}
    resp_objects = []
    for img, region, confidence in img_objs:
        if img.shape[0] <= 0 or img.shape[1] <= 0: 
            continue
        obj = {"region": region, "face_confidence": confidence}
        for action, model in models.items():
            try:
                obj.update(funcs[action](model.predict(img)))
            except Exception:
                continue

        resp_objects.append(obj)

    return dumps(resp_objects, indent=2)


def verify(img1: Union[str, np.ndarray], img2: Union[str, np.ndarray], 
           model_name: str = "VGG-Face", distance_metric: str = "cosine", 
           enforce_detection: bool = True, align: bool = True, 
           normalization: str = "base") -> Dict[str, Any]:
    """This function verifies an image pair is same person or different persons. In the background,
    verification function represents facial images as vectors and then calculates the similarity
    between those vectors. Vectors of same person images should have more similarity (or less
    distance) than vectors of different persons.

    Parameters:
            img1_path, img2_path: exact image path as string. numpy array (BGR) or based64 encoded
            images are also welcome. If one of pair has more than one face, then we will compare the
            face pair with max similarity.

            model_name (str): VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib
            , ArcFace and SFace

            distance_metric (string): cosine, euclidean, euclidean_l2

            enforce_detection (boolean): If no face could not be detected in an image, then this
            function will return exception by default. Set this to False not to have this exception.
            This might be convenient for low resolution images.

            detector_backend (string): set face detector backend to opencv, retinaface, mtcnn, ssd,
            dlib, mediapipe or yolov8.

            align (boolean): alignment according to the eye positions.

            normalization (string): normalize the input image before feeding to model

    Returns:
            Verify function returns a dictionary.

            {
                    "verified": True
                    , "distance": 0.2563
                    , "max_threshold_to_verify": 0.40
                    , "model": "VGG-Face"
                    , "similarity_metric": "cosine"
                    , 'facial_areas': {
                            'img1': {'x': 345, 'y': 211, 'w': 769, 'h': 769},
                            'img2': {'x': 318, 'y': 534, 'w': 779, 'h': 779}
                    }
                    , "time": 2
            }"""
    target_size = F.find_size(model_name)

    distances, regions = [], []
    for c1, r1, _ in F.extract_faces(img1, target_size, False, enforce_detection, align):
        for c2, r2, _ in F.extract_faces(img2, target_size, False, enforce_detection, align):
            repr1 = F.represent(c1, model_name, enforce_detection, "skip", 
                                align, normalization)[0]["embedding"]
            repr2 = F.represent(c2, model_name, enforce_detection, "skip", 
                                align, normalization)[0]["embedding"]

            if distance_metric == "cosine":
                dst = F.find_cosine_distance(repr1, repr2)
            elif distance_metric == "euclidean":
                dst = F.find_euclidean_distance(repr1, repr2)
            else:
                dst = F.find_euclidean_distance(dst.l2_normalize(repr1), 
                                                dst.l2_normalize(repr2))

            distances.append(dst)
            regions.append((r1, r2))

    threshold = F.find_threshold(model_name, distance_metric)
    distance = min(distances)
    facial_areas = regions[np.argmin(distances)]
    return {
        "verified": True if distance <= threshold else False,
        "distance": distance,
        "threshold": threshold,
        "model": model_name,
        "similarity_metric": distance_metric,
        "facial_areas": {"img1": facial_areas[0], "img2": facial_areas[1]}
    }


def get_image_metadata(image_path):
    with Image.open(image_path) as img:
        created = ctime(path.getctime(image_path))
        file_name = path.basename(image_path)
        file_size = path.getsize(image_path)
        file_access_time = ctime(path.getatime(image_path))
        
        print("Created:", created)
        print("File Name:", file_name)
        print("File Size:", file_size, "bytes")
        print("File Access Date/Time:", file_access_time)
        print("Location:", path.abspath(image_path))
        print("File Inode Change Date/Time:", ctime(path.getctime(image_path)))
        print("File Permissions:", oct(stat(image_path).st_mode)[-4:])
        print("File Type Extension:", path.splitext(image_path)[1][1:])
        print("Band names: ", img.getbands())
        print("Bounding box of the non-zero regions", img.getbbox())
        print("dpi: ", img.info["dpi"])
        print("icc_profile: ", img.info["icc_profile"])
        
        megapixels = (image_size[0] * image_size[1]) / 1000000
        print("Megapixels: ", round(megapixels, 2))
        exif_data = img.getexif()
        for tag, value in exif_data.items():
            print(f"{TAGS.get(tag, tag)}: {value}")

        return dumps({
            "image_size": img.size,
            "file_type": img.format,
            "MIME": Image.MIME[img.format],

        })

# start = time()
# a = analyze("D:/1.jpg")
# print(time() - start)
# print(a)
