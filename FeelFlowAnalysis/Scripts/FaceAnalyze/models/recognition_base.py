from abc import ABC, abstractmethod
from typing import Any, List, Union

from numpy import ndarray
from tensorflow.python.keras.models import Model


class FacialRecognitionBase(ABC):
    model: Union[Model, Any]
    model_name: str

    @abstractmethod
    def find_embeddings(self, img: ndarray) -> List[float]: ...
