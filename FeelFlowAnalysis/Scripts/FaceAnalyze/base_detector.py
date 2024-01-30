from abc import ABC, abstractmethod
from typing import List, Tuple, Union

from PIL.Image import fromarray
from numpy import arctan2, array, degrees, ndarray


class DetectorBase(ABC):
    @staticmethod
    def _align_face(img: ndarray, left_eye: Union[list, tuple], 
                   right_eye: Union[list, tuple],) -> ndarray:
        """Align a given image horizantally with respect to their left and right eye locations
        Args:
            img (np.ndarray): pre-loaded image with detected face
            left_eye (list or tuple): coordinates of left eye with respect to the you
            right_eye(list or tuple): coordinates of right eye with respect to the you
        Returns:
            img (np.ndarray): aligned facial image"""
        if (left_eye is None or right_eye is None) or \
            (img.shape[0] == 0 or img.shape[1] == 0):
            return img
        
        return array(fromarray(img).rotate(float(degrees(arctan2(
            right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])))))
    
    @abstractmethod
    def build_model(self) -> dict: pass

    @abstractmethod
    def detect_faces(self, img: ndarray, 
                     align: bool = True) -> List[Tuple[ndarray, List[float], float]]: pass
