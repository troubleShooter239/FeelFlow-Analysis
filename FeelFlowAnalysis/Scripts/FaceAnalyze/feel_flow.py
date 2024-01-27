from time import time
from typing import Any, Dict, List, Union
from json import dumps, loads

import numpy as np
from numba import jit
from cv2 import COLOR_BGR2GRAY, cvtColor, resize
from tensorflow.python.keras.models import Model

from extended_models.face_attributes import EmotionClient, GenderClient, RaceClient
from utils.modeling import build_model
from utils import functions as F


def represent(
    img_path: Union[str, np.ndarray],
    model_name: str = "VGG-Face",
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    normalization: str = "base",
) -> List[Dict[str, Any]]:
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
    resp_objs = []

    model: FacialRecognition = modeling.build_model(model_name)

    target_size = F.find_target_size(model_name=model_name)
    if detector_backend != "skip":
        img_objs = F.extract_faces(
            img=img_path,
            target_size=(target_size[1], target_size[0]),
            detector_backend=detector_backend,
            grayscale=False,
            enforce_detection=enforce_detection,
            align=align,
        )
    else:
        img, _ = F.load_image(img_path)
        if len(img.shape) == 4:
            img = img[0]  # e.g. (1, 224, 224, 3) to (224, 224, 3)
        if len(img.shape) == 3:
            img = resize(img, target_size)
            img = np.expand_dims(img, axis=0)
            if img.max() > 1:
                img = (img.astype(np.float32) / 255.0).astype(np.float32)

        img_region = {"x": 0, "y": 0, "w": img.shape[1], "h": img.shape[2]}
        img_objs = [(img, img_region, 0)]

    for img, region, confidence in img_objs:
        img = F.normalize_input(img=img, normalization=normalization)

        embedding = model.find_embeddings(img)

        resp_obj = {}
        resp_obj["embedding"] = embedding
        resp_obj["facial_area"] = region
        resp_obj["face_confidence"] = confidence
        resp_objs.append(resp_obj)

    return resp_objs


@jit(nopython=True)
def process_age(img_content: Any, model: Model) -> Dict:
    return {"age": int(Age.findApparentAge(model.predict(img_content, verbose=0)[0, :]))}


@jit(nopython=True)
def process_emotion(img_content: Any, model: Model) -> Dict:
    img_gray = resize(cvtColor(img_content[0], COLOR_BGR2GRAY), (48, 48))
    emotion = model.predict(np.expand_dims(img_gray, axis=0), verbose=0)[0, :]
    sum_predict = emotion.sum()
    return {
        "emotion": {l: round(100 * p / sum_predict, 2) for l, p in zip(EmotionClient.labels, emotion)},
        "dominant_emotion": EmotionClient.labels[np.argmax(emotion)]
    }


@jit(nopython=True)
def process_gender(img_content: Any, model: Model) -> Dict:
    gender_predict = model.predict(img_content, verbose=0)[0, :]
    return {
        "gender": {l: round(100 * p, 2) for l, p in zip(GenderClient.labels, gender_predict)},
        "dominant_gender": GenderClient.labels[np.argmax(gender_predict)]
    }         


@jit(nopython=True)
def process_race(img_content: Any, model: Model) -> Dict:
    race_predict = model.predict(img_content, verbose=0)[0, :]
    sum_predict = race_predict.sum()
    return {
        "race": {l: round(100 * p / sum_predict, 2) for l, p in zip(RaceClient.labels, race_predict)},
        "dominant_race": RaceClient.labels[np.argmax(race_predict)]
    }


def analyze(img: Union[str, np.ndarray], 
            actions: str = '{"emotion": true, "age": true, "gender": true, "race": true}',
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
    funcs = {
        "emotion": process_emotion,
        "age": process_age,
        "gender": process_gender,
        "race": process_race
    }
    items: Dict[str, bool] = loads(actions).items()
    models = {a: build_model(a.capitalize()) for a, should in items if should}

    img_objs = F.extract_faces(img, (224, 224), False, enforce_detection, align)
    
    resp_objects: List[Dict[str, Any]] = []
    for content, region, confidence in img_objs:
        if content.shape[0] < 0 or content.shape[1] < 0: 
            continue
        
        obj: Dict[str, Any] = {"region": region, "face_confidence": confidence}
        
        for action, should_analyze in items:
            if not should_analyze:
                continue
            try:
                obj.update(funcs[action](content, models[action]))
            except Exception:
                continue

        resp_objects.append(obj)

    return dumps(resp_objects, indent=2)


def verify(
    img1_path: Union[str, np.ndarray],
    img2_path: Union[str, np.ndarray],
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    distance_metric: str = "cosine",
    enforce_detection: bool = True,
    align: bool = True,
    normalization: str = "base",
) -> Dict[str, Any]:
    """
    This function verifies an image pair is same person or different persons. In the background,
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
            }

    """
    tic = time.time()

    target_size = F.find_target_size(model_name=model_name)

    img1_objs = F.extract_faces(
        img=img1_path,
        target_size=target_size,
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
    )

    img2_objs = F.extract_faces(
        img=img2_path,
        target_size=target_size,
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
    )

    distances = []
    regions = []
    for img1_content, img1_region, _ in img1_objs:
        for img2_content, img2_region, _ in img2_objs:
            img1_embedding_obj = represent(
                img_path=img1_content,
                model_name=model_name,
                enforce_detection=enforce_detection,
                detector_backend="skip",
                align=align,
                normalization=normalization,
            )

            img2_embedding_obj = represent(
                img_path=img2_content,
                model_name=model_name,
                enforce_detection=enforce_detection,
                detector_backend="skip",
                align=align,
                normalization=normalization,
            )

            img1_representation = img1_embedding_obj[0]["embedding"]
            img2_representation = img2_embedding_obj[0]["embedding"]

            if distance_metric == "cosine":
                dst = distance.findCosineDistance(img1_representation, img2_representation)
            elif distance_metric == "euclidean":
                dst = distance.findEuclideanDistance(img1_representation, img2_representation)
            else:
                dst = distance.findEuclideanDistance(
                    dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation)
                )

            distances.append(dst)
            regions.append((img1_region, img2_region))

    threshold = dst.findThreshold(model_name, distance_metric)
    dst = min(distances)
    facial_areas = regions[np.argmin(distances)]

    toc = time.time()

    resp_obj = {
        "verified": True if distance <= threshold else False,
        "distance": distance,
        "threshold": threshold,
        "model": model_name,
        "detector_backend": detector_backend,
        "similarity_metric": distance_metric,
        "facial_areas": {"img1": facial_areas[0], "img2": facial_areas[1]},
        "time": round(toc - tic, 2),
    }

    return resp_obj

start = time()
print(analyze("D:/4.jpg"))
end = time()
print(end - start)
