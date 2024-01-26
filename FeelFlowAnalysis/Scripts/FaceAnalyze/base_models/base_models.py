from abc import ABC, abstractmethod
from os.path import isfile
from typing import List, Union

from gdown import download
from tensorflow.python.keras.models import Model
from numpy import ndarray, float64


class BaseModel(ABC):
    def __init__(self) -> None:
        self.model: Model
        self.model_name: str

    @abstractmethod 
    def load_model(self, url: str) -> Model: ...

    @staticmethod
    def _download(url: str, output: str) -> None:
        if not isfile(output):
            download(url, output)


class AttributeModelBase(BaseModel):
    @abstractmethod
    def predict(self, img: ndarray) -> Union[ndarray, float64]: ...


class FacialRecognitionBase(BaseModel):
    @abstractmethod 
    def find_embeddings(self, img: ndarray) -> List[float]: ...
