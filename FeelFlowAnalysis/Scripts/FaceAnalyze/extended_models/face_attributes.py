import cv2
import numpy as np
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import (Convolution2D, Flatten, Activation, Conv2D,
    MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout)

from base_models.base_models import AttributeModelBase
from models.recognition_models import VggFaceClient
from utils.functions import get_deepface_home


class ApparentAgeClient(AttributeModelBase):
    """Age model class"""
    def __init__(self) -> None:
        self.model, self.model_name = self.load_model(), "Age"

    def predict(self, img: np.ndarray) -> np.float64:
        return np.sum(self.model.predict(img, verbose=0)[0, :] * np.array(list(range(0, 101))))

    def load_model(self, 
                   url: str = "https://github.com/serengil/deepface_models/releases/" +
                   "download/v1.0/age_model_weights.h5") -> Model:
        """Construct age model, download its weights and load
        Returns:
            model (Model)"""
        model = VggFaceClient.base_model()
        base_out = Sequential()
        base_out = Convolution2D(101, (1, 1), name="predictions")(model.layers[-4].output)
        base_out = Flatten()(base_out)
        base_out = Activation("softmax")(base_out)
        age_model = Model(inputs=model.input, outputs=base_out)
        home = get_deepface_home()
        output = home + "/.deepface/weights/age_model_weights.h5"
        self._download(url, output)
        age_model.load_weights(output)
        return age_model


class EmotionClient(AttributeModelBase):
    """Emotion model class"""
    labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    def __init__(self):
        self.model, self.model_name = self.load_model(), "Emotion"

    def predict(self, img: np.ndarray) -> np.ndarray:
        img_gray = cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, (48, 48))
        img_gray = np.expand_dims(img_gray, axis=0)
        return self.model.predict(img_gray, verbose=0)[0, :]

    def load_model(self,
                   url: str = "https://github.com/serengil/deepface_models/releases/download/" +
                   "v1.0/facial_expression_model_weights.h5") -> Sequential:
        """Construct emotion model, download and load weights"""
        num_classes = 7
        model = Sequential()
        model.add(Conv2D(64, (5, 5), activation="relu", input_shape=(48, 48, 1)))
        model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Conv2D(128, (3, 3), activation="relu"))
        model.add(Conv2D(128, (3, 3), activation="relu"))
        model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1024, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(1024, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation="softmax"))
        home = get_deepface_home()
        output = home + "/.deepface/weights/facial_expression_model_weights.h5"
        self._download(url, output)
        model.load_weights(output)
        return model


class GenderClient(AttributeModelBase):
    """Gender model class"""
    labels = ["Woman", "Man"]

    def __init__(self):
        self.model, self.model_name = self.load_model(), "Gender"

    def predict(self, img: np.ndarray) -> np.ndarray:
        return self.model.predict(img, verbose=0)[0, :]

    def load_model(self, url: str = "https://github.com/serengil/deepface_models/releases/"
                   "download/v1.0/gender_model_weights.h5",) -> Model:
        """Construct gender model, download its weights and load
        Returns:
            model (Model)"""
        model = VggFaceClient.base_model()
        base_model_output = Sequential()
        base_model_output = Convolution2D(2, (1, 1), name="predictions")(model.layers[-4].output)
        base_model_output = Flatten()(base_model_output)
        base_model_output = Activation("softmax")(base_model_output)
        gender_model = Model(inputs=model.input, outputs=base_model_output)
        home = get_deepface_home()
        output = home + "/.deepface/weights/gender_model_weights.h5"
        self._download(url, output)
        gender_model.load_weights(output)
        return gender_model


class RaceClient(AttributeModelBase):
    """Race model class"""
    labels = ["asian", "indian", "black", "white", "middle eastern", "latino hispanic"]

    def __init__(self):
        self.model, self.model_name = self.load_model(), "Race"

    def predict(self, img: np.ndarray) -> np.ndarray:
        return self.model.predict(img, verbose=0)[0, :]

    def load_model(self, url: str = "https://github.com/serengil/deepface_models/releases/" +
                   "download/v1.0/race_model_single_batch.h5") -> Model:
        """Construct race model, download its weights and load"""
        model = VggFaceClient.base_model()
        base_model_output = Sequential()
        base_model_output = Convolution2D(6, (1, 1), name="predictions")(model.layers[-4].output)
        base_model_output = Flatten()(base_model_output)
        base_model_output = Activation("softmax")(base_model_output)
        race_model = Model(inputs=model.input, outputs=base_model_output)
        home = get_deepface_home()
        output = home + "/.deepface/weights/race_model_single_batch.h5"
        self._download(url, output)
        race_model.load_weights(output)
        return race_model
