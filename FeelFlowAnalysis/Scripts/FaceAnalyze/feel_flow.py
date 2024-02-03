from os import path, stat
from time import ctime, time
from typing import Any, Dict, List, Tuple, Union
from json import dumps, loads

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
from PIL.TiffImagePlugin import IFDRational

import utils.functions as F
from detectors.opencv_client import DetectorWrapper
from loaders.image_loader import load_image
from utils.distance import find_cosine, find_euclidean
from utils.modeling import build_model
from models.face_attributes import EmotionClient, GenderClient, RaceClient

if F.get_tf_major_version() == 1:
    from keras.models import Model
    from keras.preprocessing import image
else:
    from tensorflow.keras.models import Model
    from tensorflow.keras.preprocessing import image

F.initialize_folder()


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
    """This function analyzes facial attributes including age, gender, emotion and race.
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
            ]"""
    funcs = {"age": process_age, "emotion": process_emotion, "gender": process_gender, "race": process_race}
    try:
        img_objs = extract_faces(img, (224, 224), False, enforce_detection, align)
    except ValueError:
        return "{}"
    models: Dict[str, Model] = {a: build_model(a.capitalize()) for a, s in loads(actions).items() if s}
    resp_objects = []
    # TODO: Make it parallel
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


def represent(img_path: Union[str, np.ndarray], model_name: str = "VGG-Face", 
              enforce_detection: bool = True, detector_backend: str = "opencv",
              align: bool = True, normalization: str = "base") -> List[Dict[str, Any]]:
    """Represent facial images as multi-dimensional vector embeddings.

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
            to 'skip', the confidence will be 0 and is nonsensical."""
    model = build_model(model_name)
    target_size = F.find_size(model_name)
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
        e = model.find_embeddings(F.normalize_input(i, normalization))
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

        img_pixels = np.expand_dims(image.img_to_array(current_img), axis=0) / 255
        regs = {"x": int(reg[0]), "y": int(reg[1]), "w": int(reg[2]), "h": int(reg[3])}
        extracted_faces.append((img_pixels, regs, confidence))

    if len(extracted_faces) == 0 and enforce_detection:
        raise ValueError(f"Detected face shape is {img.shape}. "
                         "Consider to set enforce_detection arg to False.")

    return extracted_faces


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
    for c1, r1, _ in extract_faces(img1, target_size, False, enforce_detection, align):
        for c2, r2, _ in extract_faces(img2, target_size, False, enforce_detection, align):
            repr1 = represent(c1, model_name, enforce_detection, "skip", 
                              align, normalization)[0]["embedding"]
            repr2 = represent(c2, model_name, enforce_detection, "skip", 
                              align, normalization)[0]["embedding"]

            if distance_metric == "cosine":
                dst = find_cosine(repr1, repr2)
            elif distance_metric == "euclidean":
                dst = find_euclidean(repr1, repr2)
            else:
                dst = find_euclidean(dst.l2_normalize(repr1), dst.l2_normalize(repr2))

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


# def get_gps(exif_data) -> Tuple:
#     if 'GPSInfo' in exif_data:
#         gps_info = exif_data['GPSInfo']
#         lat_ref = gps_info[1]
#         lat = gps_info[2]
#         lon_ref = gps_info[3]
#         lon = gps_info[4]
#         lat_deg = lat[0][0] / float(lat[0][1])
#         lat_min = lat[1][0] / float(lat[1][1])
#         lat_sec = lat[2][0] / float(lat[2][1])
#         lon_deg = lon[0][0] / float(lon[0][1])
#         lon_min = lon[1][0] / float(lon[1][1])
#         lon_sec = lon[2][0] / float(lon[2][1])
#         if lat_ref == 'S':
#             lat_deg = -lat_deg
#         if lon_ref == 'W':
#             lon_deg = -lon_deg
#         return lat_deg, lat_min, lat_sec, lon_deg, lon_min, lon_sec
#     return ()
# from PIL import Image
# import piexif


# img = Image.open("D:/1.jpg")
# exif_dict = piexif.load(img.info['exif'])
# piexif.GPSIFD.GPSLatitude
# if 'GPS' in exif_dict:
#     gps = exif_dict['GPS']
#     print(gps)
#     latitude = gps[2][0][0] / float(gps[2][0][1])
#     longitude = gps[4][0][0] / float(gps[4][0][1])
#     print('Широта:', latitude)
#     print('Долгота:', longitude)
# else:
#     print('Геоданные отсутствуют')


# TODO: Converting to ndarray is bring's losing img information
def get_image_metadata(image: Union[str, np.ndarray]) -> str:
    with Image.open(image) as i:
        try:
            mime = Image.MIME[i.format]
        except KeyError:
            mime = None
        data = {
            "image_size": i.size,
            "file_type": i.format,
            "mime": mime,
            #"time_created": ctime(path.getctime(i)),
            #"name": path.basename(i),
            #"size": path.getsize(i),
            #"access_time": ctime(path.getatime(i)),
            #"location": path.abspath(i),
            #"inode_change_time": ctime(path.getctime(i)),
            #"permission": oct(stat(i).st_mode)[-4:],
            #"type_extension": path.splitext(i)[1][1:],
            "band_names": i.getbands(),
            "bbox": i.getbbox(),
            #"icc_profile": i.info["icc_profile"],
            "megapixels": round(i.size[0] * i.size[1] / 1000000, 2),
        }
        data["dpi"] = tuple(map(float, i.info["dpi"]))
        exif_data = {TAGS.get(t, t): float(v) if isinstance(v, IFDRational) else v for t, v in i.getexif().items()}
        data.update(exif_data)
        data["GPSInfo"] = get_gps(exif_data)
    
    return dumps(data, indent=2)


# print(get_image_metadata("D:/1.jpg"))

# #start = time()
# a = analyze("D:/1.jpg")
# #print(time() - start)
# print(a)
