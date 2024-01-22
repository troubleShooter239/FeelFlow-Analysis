from collections import defaultdict
from typing import Any, Dict, Union
from json import dumps, loads

import cv2
import numpy as np
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
from deepface.commons.functions import extract_faces
from deepface.extendedmodels import Age, Gender, Race, Emotion
from keras.models import Model


def build_model(model_name: str) -> Union[Model, Any]:
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
        model = models.get(model_name)
        model = model()
        model_obj[model_name] = model

    return model_obj[model_name]


def process_emotion(img_content: Any, model: Model) -> Dict:
    img_gray = cv2.cvtColor(img_content, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (48, 48))
    img_gray = np.expand_dims(img_gray, axis=0)

    emotion = model.predict(img_gray, verbose=0)[0, :]
    sum_predict = emotion.sum()

    return {
        "emotion": {l: round(100 * p / sum_predict, 2) for l, p in zip(Emotion.labels, emotion)},
        "dominant_emotion": Emotion.labels[np.argmax(emotion)]
    }


def process_age(img_content: Any, model: Model) -> Dict:
    return {"age": int(Age.findApparentAge(model.predict(img_content, verbose=0)[0, :]))}


def process_gender(img_content: Any, model: Model) -> Dict:
    gender_predict = model.predict(img_content, verbose=0)[0, :]
    return {
        "gender": {l: round(100 * p, 2) for l, p in zip(Gender.labels, gender_predict)},
        "dominant_gender": Gender.labels[np.argmax(gender_predict)]
    }         


def process_race(img_content: Any, model: Model) -> Dict:
    race_predict = model.predict(img_content, verbose=0)[0, :]
    sum_predict = race_predict.sum()
    return {
        "race": {l: round(100 * p / sum_predict, 2) for l, p in zip(Race.labels, race_predict)},
        "dominant_race": Race.labels[np.argmax(race_predict)]
    }


def analyze(
    img: Union[str, np.ndarray],
    actions: str =  '{"emotion": true, "age": true, "gender": true, "race": true}',
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
    
    resp_objects = []

    img_objs = extract_faces(
        img=img,
        target_size=(224, 224),
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align
    )

    for img_content, img_region, img_confidence in img_objs:
        if not(img_content.shape[0] > 0 and img_content.shape[1] > 0): 
            continue
        
        obj = {"region": img_region, "face_confidence": img_confidence}
        
        for action, should_analyze in actions.items():
            if should_analyze: 
                try:
                    obj.update(funcs[action](img_content, models[action]))
                except Exception:
                    pass

        resp_objects.append(obj)

    return dumps(resp_objects, indent=2)