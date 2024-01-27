from os.path import isfile, sep
from typing import Any, List, Tuple

import cv2
from numpy import ndarray

from base_detector.base_detector import DetectorBase


class OpenCvClient(DetectorBase):
    """Class to cover common face detection functionalitiy for OpenCv backend"""
    def __init__(self):
        self.model = self.build_model()

    def build_model(self) -> dict:
        """Build opencv's face and eye detector models
        Returns:
            model (dict): including face_detector and eye_detector keys"""
        return {
            "face_detector": self.__build_cascade("haarcascade"),
            "eye_detector": self.__build_cascade("haarcascade_eye")
        }

    def detect_faces(self, img: ndarray, 
                     align: bool = True) -> List[Tuple[ndarray, List[float], float]]:
        """Detect and align face with opencv
        Args:
            face_detector (Any): opencv face detector object
            img (np.ndarray): pre-loaded image
            align (bool): default is true
        Returns:
            results (List[Tuple[np.ndarray, List[float], float]]): A list of tuples
                where each tuple contains:
                - detected_face (np.ndarray): The detected face as a NumPy array.
                - face_region (List[float]): The image region represented as
                    a list of floats e.g. [x, y, w, h]
                - confidence (float): The confidence score associated with the detected face.

        Example:
            results = [
                (array(..., dtype=uint8), [110, 60, 150, 380], 0.99),
                (array(..., dtype=uint8), [150, 50, 299, 375], 0.98),
                (array(..., dtype=uint8), [120, 55, 300, 371], 0.96),
            ]"""
        img_region = [0, 0, img.shape[1], img.shape[0]]
        faces = []
        
        try:
            faces, _, scores = self.model["face_detector"].detectMultiScale3(
                img, 1.1, 10, outputRejectLevels=True
            )
        except:
            pass

        if len(faces) == 0:
            return []

        resp = []
        for (x, y, w, h), confidence in zip(faces, scores):
            detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]

            if align:
                left_eye, right_eye = self.find_eyes(detected_face)
                detected_face = self._align_face(detected_face, left_eye, right_eye)

            img_region = [x, y, w, h]

            resp.append((detected_face, img_region, confidence))

        return resp

    def find_eyes(self, img: ndarray) -> tuple:
        """Find the left and right eye coordinates of given image
        Args:
            img (np.ndarray): given image
        Returns:
            left and right eye (tuple)"""
        left_eye = None
        right_eye = None

        if img.shape[0] == 0 or img.shape[1] == 0:
            return left_eye, right_eye

        eyes = self.model["eye_detector"].detectMultiScale(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.1, 10)
        eyes = sorted(eyes, key=lambda v: abs(v[2] * v[3]), reverse=True)

        if len(eyes) >= 2:
            eye_1 = eyes[0]
            eye_2 = eyes[1]

            if eye_1[0] < eye_2[0]:
                left_eye = eye_1
                right_eye = eye_2
            else:
                left_eye = eye_2
                right_eye = eye_1

            left_eye = (int(left_eye[0] + (left_eye[2] / 2)), 
                        int(left_eye[1] + (left_eye[3] / 2)))
            right_eye = (int(right_eye[0] + (right_eye[2] / 2)),
                         int(right_eye[1] + (right_eye[3] / 2)))
            
        return left_eye, right_eye

    def __build_cascade(self, model_name="haarcascade") -> Any:
        """Build a opencv face&eye detector models
        Returns:
            model (Any)"""
        opencv_path = self.__get_opencv_path()
        if model_name == "haarcascade":
            face_detector_path = opencv_path + "haarcascade_frontalface_default.xml"
            if not isfile(face_detector_path):
                raise ValueError(
                    "Confirm that opencv is installed on your environment! Expected path ",
                    face_detector_path,
                    " violated.",
                )
            detector = cv2.CascadeClassifier(face_detector_path)

        elif model_name == "haarcascade_eye":
            eye_detector_path = opencv_path + "haarcascade_eye.xml"
            if not isfile(eye_detector_path):
                raise ValueError(
                    "Confirm that opencv is installed on your environment! Expected path ",
                    eye_detector_path,
                    " violated."
                )
            detector = cv2.CascadeClassifier(eye_detector_path)

        else:
            raise ValueError(f"unimplemented model_name for build_cascade - {model_name}")

        return detector

    def __get_opencv_path(self) -> str:
        """Returns where opencv installed
        Returns:
            installation_path (str)"""
        folders = cv2.__file__.split(sep)[0:-1]
        path = folders[0]
        for folder in folders[1:]:
            path = path + "/" + folder

        return path + "/data/"


class DetectorWrapper:
    @staticmethod
    def build_model() -> OpenCvClient:
        """Build a face detector model
        Args:
            detector_backend (str): backend detector name
        Returns:
            built detector (Any)"""
        global face_detector_obj
        if not "face_detector_obj" in globals():
            face_detector_obj = {}
        detector_backend = "opencv"
        built_models = list(face_detector_obj.keys())
        if detector_backend not in built_models:
            face_detector_obj[detector_backend] = OpenCvClient()            
        return face_detector_obj[detector_backend]

    @staticmethod
    def detect_faces(img: ndarray, align: bool = True) -> list:
        """Detect face(s) from a given image
        Args:
            detector_backend (str): detector name
            img (np.ndarray): pre-loaded image
            alig (bool): enable or disable alignment after detection
        Returns:
            results (List[Tuple[np.ndarray, List[float], float]]): A list of tuples
                where each tuple contains:
                - detected_face (np.ndarray): The detected face as a NumPy array.
                - face_region (List[float]): The image region represented as
                    a list of floats e.g. [x, y, w, h]
                - confidence (float): The confidence score associated with the detected face.

        Example:
            results = [
                (array(..., dtype=uint8), [110, 60, 150, 380], 0.99),
                (array(..., dtype=uint8), [150, 50, 299, 375], 0.98),
                (array(..., dtype=uint8), [120, 55, 300, 371], 0.96),
            ]"""
        return DetectorWrapper.build_model().detect_faces(img=img, align=align)
