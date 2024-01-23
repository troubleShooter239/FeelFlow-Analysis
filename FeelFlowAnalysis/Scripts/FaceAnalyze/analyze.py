from collections import defaultdict
from time import time
from typing import Any, Dict, List, Union
from json import dumps, loads

import numpy as np
from numba import jit
from cv2 import COLOR_BGR2GRAY, cvtColor, resize
from deepface.basemodels import (
    VGGFace,
    OpenFace,
    Facenet,
    Facenet512,
    FbDeepFace,
    DeepID,
    DlibWrapper,
    ArcFace,
    SFace,
)
from deepface.commons import distance, functions
from deepface.extendedmodels import Age, Gender, Race, Emotion
from keras.models import Model


def build_model(model_name: str) -> Model:
    """
    This function builds a deepface model
    Parameters:
            model_name (string): face recognition or facial attribute model
                    VGG-Face, Facenet, OpenFace, DeepFace, DeepID for face recognition
                    Age, Gender, Emotion, Race for facial attributes

    Returns:
            built deepface model ( (tf.)keras.models.Model )
    """
    global model_obj
    model_obj = defaultdict(dict)

    models = {
        "VGG-Face": VGGFace.loadModel,
        "OpenFace": OpenFace.loadModel,
        "Facenet": Facenet.loadModel,
        "Facenet512": Facenet512.loadModel,
        "DeepFace": FbDeepFace.loadModel,
        "DeepID": DeepID.loadModel,
        "Dlib": DlibWrapper.loadModel,
        "ArcFace": ArcFace.loadModel,
        "SFace": SFace.load_model,
        "Emotion": Emotion.loadModel,
        "Age": Age.loadModel,
        "Gender": Gender.loadModel,
        "Race": Race.loadModel,
    }

    if model_name not in model_obj[model_name]:
        model_obj[model_name] = models.get(model_name)()

    return model_obj[model_name]


def represent(
    img_path: Union[str, np.ndarray],
    model_name: str = "VGG-Face",
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    normalization: str = "base",
) -> List[Dict[str, Any]]:
    """
    This function represents facial images as vectors. The function uses convolutional neural
    networks models to generate vector embeddings.

    Parameters:
            img_path (string): exact image path. Alternatively, numpy array (BGR) or based64
            encoded images could be passed. Source image can have many faces. Then, result will
            be the size of number of faces appearing in the source image.

            model_name (string): VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib,
            ArcFace, SFace

            enforce_detection (boolean): If no face could not be detected in an image, then this
            function will return exception by default. Set this to False not to have this exception.
            This might be convenient for low resolution images.

            detector_backend (string): set face detector backend to opencv, retinaface, mtcnn, ssd,
            dlib, mediapipe or yolov8. A special value `skip` could be used to skip face-detection
            and only encode the given image.

            align (boolean): alignment according to the eye positions.

            normalization (string): normalize the input image before feeding to model

    Returns:
            Represent function returns a list of object, each object has fields as follows:
            {
                "embedding": np.array,
                "facial_area": dict{"x": int, "y": int, "w": int, "h": int},
                "face_confidence": float
            }
    """
    resp_objs = []

    model = build_model(model_name)

    target_size = functions.find_target_size(model_name=model_name)
    if detector_backend != "skip":
        img_objs = functions.extract_faces(
            img=img_path,
            target_size=target_size,
            detector_backend=detector_backend,
            grayscale=False,
            enforce_detection=enforce_detection,
            align=align,
        )
    else:
        img, _ = functions.load_image(img_path)
        if len(img.shape) == 4:
            img = img[0]  # e.g. (1, 224, 224, 3) to (224, 224, 3)
        if len(img.shape) == 3:
            img = np.expand_dims(resize(img, target_size), axis=0)
            if img.max() > 1:
                img = img.astype(np.float32) / 255.0

        img_region = {"x": 0, "y": 0, "w": img.shape[1], "h": img.shape[2]}
        img_objs = [(img, img_region, 0)]

    for img, region, confidence in img_objs:
        # custom normalization
        img = functions.normalize_input(img=img, normalization=normalization)

        # represent
        # if "keras" in str(type(model)):
        if isinstance(model, Model):
            # model.predict causes memory issue when it is called in a for loop
            # embedding = model.predict(img, verbose=0)[0].tolist()
            embedding = model(img, training=False).numpy()[0].tolist()
            # if you still get verbose logging. try call
            # - `tf.keras.utils.disable_interactive_logging()`
            # in your main program
        else:
            # SFace and Dlib are not keras models and no verbose arguments
            embedding = model.predict(img)[0].tolist()

        resp_obj = {}
        resp_obj["embedding"] = embedding
        resp_obj["facial_area"] = region
        resp_obj["face_confidence"] = confidence
        resp_objs.append(resp_obj)

    return resp_objs


@jit(nopython=True)
def process_emotion(img_content: Any, model: Model) -> Dict:
    img_gray = resize(cvtColor(img_content[0], COLOR_BGR2GRAY), (48, 48))
    emotion = model.predict(np.expand_dims(img_gray, axis=0), verbose=0)[0, :]
    sum_predict = emotion.sum()
    return {
        "emotion": {l: round(100 * p / sum_predict, 2) for l, p in zip(Emotion.labels, emotion)},
        "dominant_emotion": Emotion.labels[np.argmax(emotion)]
    }


@jit(nopython=True)
def process_age(img_content: Any, model: Model) -> Dict:
    return {"age": int(Age.findApparentAge(model.predict(img_content, verbose=0)[0, :]))}


@jit(nopython=True)
def process_gender(img_content: Any, model: Model) -> Dict:
    gender_predict = model.predict(img_content, verbose=0)[0, :]
    return {
        "gender": {l: round(100 * p, 2) for l, p in zip(Gender.labels, gender_predict)},
        "dominant_gender": Gender.labels[np.argmax(gender_predict)]
    }         


@jit(nopython=True)
def process_race(img_content: Any, model: Model) -> Dict:
    race_predict = model.predict(img_content, verbose=0)[0, :]
    sum_predict = race_predict.sum()
    return {
        "race": {l: round(100 * p / sum_predict, 2) for l, p in zip(Race.labels, race_predict)},
        "dominant_race": Race.labels[np.argmax(race_predict)]
    }


def analyze(
    img: Union[str, np.ndarray],
    actions: str = '{"emotion": true, "age": true, "gender": true, "race": true}',
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
) -> str:
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
    
    actions = loads(actions)
    models = {act: build_model(act.capitalize()) for act, should_build in actions.items() if should_build}

    img_objs = functions.extract_faces(
        img=img,
        target_size=(224, 224),
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align
    )

    resp_objects = []

    for content, region, confidence in img_objs:
        if content.shape[0] < 0 or content.shape[1] < 0: 
            continue
        
        obj = {"region": region, "face_confidence": confidence}
        
        for action, should_analyze in actions.items():
            if should_analyze: 
                try:
                    obj.update(funcs[action](content, models[action]))
                except Exception:
                    pass

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

    target_size = functions.find_target_size(model_name=model_name)

    img1_objs = functions.extract_faces(
        img=img1_path,
        target_size=target_size,
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
    )

    img2_objs = functions.extract_faces(
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
analyze("C:/Users/morgu/Desktop/3.jpg")
end = time()
print(end - start)
