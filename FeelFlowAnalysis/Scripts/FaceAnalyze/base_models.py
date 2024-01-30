from abc import ABC, abstractmethod
from os.path import isfile
from typing import List, Union

from gdown import download
from tensorflow.keras.backend import int_shape
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Activation, Input, Conv2D, 
    BatchNormalization, MaxPooling2D, Dropout, Dense, Concatenate, Lambda, 
    GlobalAveragePooling2D, add)
from numpy import ndarray, float64

from constants import DOWNLOAD_URL
from functions import get_deepface_home


class BaseModel(ABC):
    def __init__(self) -> None:
        self.model: Model
        self.model_name: str

    @abstractmethod 
    def load_model(self, url: str) -> Model: pass

    @staticmethod
    def _download(url: str, output: str) -> None:
        if not isfile(output):
            download(url, output)


class AttributeModelBase(BaseModel):
    @abstractmethod
    def predict(self, img: ndarray) -> Union[ndarray, float64]: pass


class FacialRecognitionBase(BaseModel):
    @abstractmethod 
    def find_embeddings(self, img: ndarray) -> List[float]: pass


class FaceNetBase(FacialRecognitionBase):
    @staticmethod
    def _inception_res_netV2(dimension: int = 128) -> Model:
        """InceptionResNetV2 model
        Args:
            dimension (int): number of dimensions in the embedding layer
        Returns:
            model (Model)"""
        def scaling(x, scale): return x * scale
        
        inputs = Input(shape=(160, 160, 3))
        x = Conv2D(32, 3, strides=2, padding="valid", use_bias=False, name="Conv2d_1a_3x3")(inputs)
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                            name="Conv2d_1a_3x3_BatchNorm")(x)
        x = Activation("relu", name="Conv2d_1a_3x3_Activation")(x)
        x = Conv2D(32, 3, strides=1, padding="valid", use_bias=False, name="Conv2d_2a_3x3")(x)
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                            name="Conv2d_2a_3x3_BatchNorm")(x)
        x = Activation("relu", name="Conv2d_2a_3x3_Activation")(x)
        x = Conv2D(64, 3, strides=1, padding="same", use_bias=False, name="Conv2d_2b_3x3")(x)
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                            name="Conv2d_2b_3x3_BatchNorm")(x)
        x = Activation("relu", name="Conv2d_2b_3x3_Activation")(x)
        x = MaxPooling2D(3, strides=2, name="MaxPool_3a_3x3")(x)
        x = Conv2D(80, 1, strides=1, padding="valid", use_bias=False, name="Conv2d_3b_1x1")(x)
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                            name="Conv2d_3b_1x1_BatchNorm")(x)
        x = Activation("relu", name="Conv2d_3b_1x1_Activation")(x)
        x = Conv2D(192, 3, strides=1, padding="valid", use_bias=False, name="Conv2d_4a_3x3")(x)
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                            name="Conv2d_4a_3x3_BatchNorm")(x)
        x = Activation("relu", name="Conv2d_4a_3x3_Activation")(x)
        x = Conv2D(256, 3, strides=2, padding="valid", use_bias=False, name="Conv2d_4b_3x3")(x)
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                            name="Conv2d_4b_3x3_BatchNorm")(x)
        x = Activation("relu", name="Conv2d_4b_3x3_Activation")(x)
        branch_0 = Conv2D(32, 1, strides=1, padding="same", use_bias=False, 
                        name="Block35_1_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block35_1_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block35_1_Branch_0_Conv2d_1x1_Activation")(branch_0)
        branch_1 = Conv2D(32, 1, strides=1, padding="same", use_bias=False, 
                        name="Block35_1_Branch_1_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block35_1_Branch_1_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block35_1_Branch_1_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(32, 3, strides=1, padding="same", use_bias=False, 
                        name="Block35_1_Branch_1_Conv2d_0b_3x3")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block35_1_Branch_1_Conv2d_0b_3x3_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block35_1_Branch_1_Conv2d_0b_3x3_Activation")(branch_1)
        branch_2 = Conv2D(32, 1, strides=1, padding="same", use_bias=False, 
                        name="Block35_1_Branch_2_Conv2d_0a_1x1")(x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block35_1_Branch_2_Conv2d_0a_1x1_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Block35_1_Branch_2_Conv2d_0a_1x1_Activation")(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding="same", use_bias=False, 
                        name="Block35_1_Branch_2_Conv2d_0b_3x3")(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block35_1_Branch_2_Conv2d_0b_3x3_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Block35_1_Branch_2_Conv2d_0b_3x3_Activation")(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding="same", use_bias=False, 
                        name="Block35_1_Branch_2_Conv2d_0c_3x3")(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block35_1_Branch_2_Conv2d_0c_3x3_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Block35_1_Branch_2_Conv2d_0c_3x3_Activation")(branch_2)
        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(axis=3, name="Block35_1_Concatenate")(branches)
        up = Conv2D(256, 1, strides=1, padding="same", use_bias=True, 
                    name="Block35_1_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.17})(up)
        x = add([x, up])
        x = Activation("relu", name="Block35_1_Activation")(x)
        branch_0 = Conv2D(32, 1, strides=1, padding="same", use_bias=False, 
                        name="Block35_2_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block35_2_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block35_2_Branch_0_Conv2d_1x1_Activation")(branch_0)
        branch_1 = Conv2D(32, 1, strides=1, padding="same", use_bias=False, 
                        name="Block35_2_Branch_1_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block35_2_Branch_1_Conv2d_0a_1x1_BatchNorm",)(branch_1)
        branch_1 = Activation("relu", name="Block35_2_Branch_1_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(32, 3, strides=1, padding="same", use_bias=False, 
                        name="Block35_2_Branch_1_Conv2d_0b_3x3")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block35_2_Branch_1_Conv2d_0b_3x3_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block35_2_Branch_1_Conv2d_0b_3x3_Activation")(branch_1)
        branch_2 = Conv2D(32, 1, strides=1, padding="same", use_bias=False, 
                        name="Block35_2_Branch_2_Conv2d_0a_1x1")(x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block35_2_Branch_2_Conv2d_0a_1x1_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Block35_2_Branch_2_Conv2d_0a_1x1_Activation")(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding="same", use_bias=False, 
                        name="Block35_2_Branch_2_Conv2d_0b_3x3")(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                      name="Block35_2_Branch_2_Conv2d_0b_3x3_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Block35_2_Branch_2_Conv2d_0b_3x3_Activation")(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding="same", use_bias=False, 
                        name="Block35_2_Branch_2_Conv2d_0c_3x3")(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block35_2_Branch_2_Conv2d_0c_3x3_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Block35_2_Branch_2_Conv2d_0c_3x3_Activation")(branch_2)
        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(axis=3, name="Block35_2_Concatenate")(branches)
        up = Conv2D(256, 1, strides=1, padding="same", use_bias=True, 
                    name="Block35_2_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.17})(up)
        x = add([x, up])
        x = Activation("relu", name="Block35_2_Activation")(x)
        branch_0 = Conv2D(32, 1, strides=1, padding="same", use_bias=False,
                        name="Block35_3_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block35_3_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block35_3_Branch_0_Conv2d_1x1_Activation")(branch_0)
        branch_1 = Conv2D(32, 1, strides=1, padding="same", use_bias=False, 
                        name="Block35_3_Branch_1_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block35_3_Branch_1_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block35_3_Branch_1_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(32, 3, strides=1, padding="same", use_bias=False, 
                        name="Block35_3_Branch_1_Conv2d_0b_3x3")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block35_3_Branch_1_Conv2d_0b_3x3_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block35_3_Branch_1_Conv2d_0b_3x3_Activation")(branch_1)
        branch_2 = Conv2D(32, 1, strides=1, padding="same", use_bias=False, 
                        name="Block35_3_Branch_2_Conv2d_0a_1x1")(x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block35_3_Branch_2_Conv2d_0a_1x1_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Block35_3_Branch_2_Conv2d_0a_1x1_Activation")(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding="same", use_bias=False, 
                        name="Block35_3_Branch_2_Conv2d_0b_3x3")(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block35_3_Branch_2_Conv2d_0b_3x3_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Block35_3_Branch_2_Conv2d_0b_3x3_Activation")(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding="same", use_bias=False, 
                        name="Block35_3_Branch_2_Conv2d_0c_3x3")(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block35_3_Branch_2_Conv2d_0c_3x3_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Block35_3_Branch_2_Conv2d_0c_3x3_Activation")(branch_2)
        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(axis=3, name="Block35_3_Concatenate")(branches)
        up = Conv2D(256, 1, strides=1, padding="same", use_bias=True, 
                    name="Block35_3_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.17})(up)
        x = add([x, up])
        x = Activation("relu", name="Block35_3_Activation")(x)
        branch_0 = Conv2D(32, 1, strides=1, padding="same", use_bias=False, 
                        name="Block35_4_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block35_4_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block35_4_Branch_0_Conv2d_1x1_Activation")(branch_0)
        branch_1 = Conv2D(32, 1, strides=1, padding="same", use_bias=False, 
                        name="Block35_4_Branch_1_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block35_4_Branch_1_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block35_4_Branch_1_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(32, 3, strides=1, padding="same", use_bias=False, 
                        name="Block35_4_Branch_1_Conv2d_0b_3x3")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block35_4_Branch_1_Conv2d_0b_3x3_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block35_4_Branch_1_Conv2d_0b_3x3_Activation")(branch_1)
        branch_2 = Conv2D(32, 1, strides=1, padding="same", use_bias=False, 
                        name="Block35_4_Branch_2_Conv2d_0a_1x1")(x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                    name="Block35_4_Branch_2_Conv2d_0a_1x1_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Block35_4_Branch_2_Conv2d_0a_1x1_Activation")(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding="same", use_bias=False, 
                        name="Block35_4_Branch_2_Conv2d_0b_3x3")(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block35_4_Branch_2_Conv2d_0b_3x3_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Block35_4_Branch_2_Conv2d_0b_3x3_Activation")(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding="same", use_bias=False, 
                        name="Block35_4_Branch_2_Conv2d_0c_3x3")(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block35_4_Branch_2_Conv2d_0c_3x3_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Block35_4_Branch_2_Conv2d_0c_3x3_Activation")(branch_2)
        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(axis=3, name="Block35_4_Concatenate")(branches)
        up = Conv2D(256, 1, strides=1, padding="same", use_bias=True, 
                    name="Block35_4_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.17})(up)
        x = add([x, up])
        x = Activation("relu", name="Block35_4_Activation")(x)
        branch_0 = Conv2D(32, 1, strides=1, padding="same", use_bias=False, 
                        name="Block35_5_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block35_5_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block35_5_Branch_0_Conv2d_1x1_Activation")(branch_0)
        branch_1 = Conv2D(32, 1, strides=1, padding="same", use_bias=False, 
                        name="Block35_5_Branch_1_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block35_5_Branch_1_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block35_5_Branch_1_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(32, 3, strides=1, padding="same", use_bias=False, 
                        name="Block35_5_Branch_1_Conv2d_0b_3x3")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block35_5_Branch_1_Conv2d_0b_3x3_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block35_5_Branch_1_Conv2d_0b_3x3_Activation")(branch_1)
        branch_2 = Conv2D(32, 1, strides=1, padding="same", use_bias=False, 
                        name="Block35_5_Branch_2_Conv2d_0a_1x1")(x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block35_5_Branch_2_Conv2d_0a_1x1_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Block35_5_Branch_2_Conv2d_0a_1x1_Activation")(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding="same", use_bias=False, 
                        name="Block35_5_Branch_2_Conv2d_0b_3x3")(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block35_5_Branch_2_Conv2d_0b_3x3_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Block35_5_Branch_2_Conv2d_0b_3x3_Activation")(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding="same", use_bias=False, 
                        name="Block35_5_Branch_2_Conv2d_0c_3x3")(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block35_5_Branch_2_Conv2d_0c_3x3_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Block35_5_Branch_2_Conv2d_0c_3x3_Activation")(branch_2)
        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(axis=3, name="Block35_5_Concatenate")(branches)
        up = Conv2D(256, 1, strides=1, padding="same", use_bias=True, 
                    name="Block35_5_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.17})(up)
        x = add([x, up])
        x = Activation("relu", name="Block35_5_Activation")(x)
        branch_0 = Conv2D(384, 3, strides=2, padding="valid", use_bias=False, 
                        name="Mixed_6a_Branch_0_Conv2d_1a_3x3")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Mixed_6a_Branch_0_Conv2d_1a_3x3_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Mixed_6a_Branch_0_Conv2d_1a_3x3_Activation")(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding="same", use_bias=False, 
                        name="Mixed_6a_Branch_1_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Mixed_6a_Branch_1_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Mixed_6a_Branch_1_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(192, 3, strides=1, padding="same", use_bias=False, 
                        name="Mixed_6a_Branch_1_Conv2d_0b_3x3")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Mixed_6a_Branch_1_Conv2d_0b_3x3_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Mixed_6a_Branch_1_Conv2d_0b_3x3_Activation")(branch_1)
        branch_1 = Conv2D(256, 3, strides=2, padding="valid", use_bias=False, 
                        name="Mixed_6a_Branch_1_Conv2d_1a_3x3")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Mixed_6a_Branch_1_Conv2d_1a_3x3_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Mixed_6a_Branch_1_Conv2d_1a_3x3_Activation")(branch_1)
        branch_pool = MaxPooling2D(3, strides=2, padding="valid", 
                                name="Mixed_6a_Branch_2_MaxPool_1a_3x3")(x)
        branches = [branch_0, branch_1, branch_pool]
        x = Concatenate(axis=3, name="Mixed_6a")(branches)
        branch_0 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, 
                        name="Block17_1_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_1_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block17_1_Branch_0_Conv2d_1x1_Activation")(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, 
                        name="Block17_1_Branch_1_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_1_Branch_1_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_1_Branch_1_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding="same", use_bias=False, 
                        name="Block17_1_Branch_1_Conv2d_0b_1x7")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_1_Branch_1_Conv2d_0b_1x7_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_1_Branch_1_Conv2d_0b_1x7_Activation")(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding="same", use_bias=False, 
                        name="Block17_1_Branch_1_Conv2d_0c_7x1")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_1_Branch_1_Conv2d_0c_7x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_1_Branch_1_Conv2d_0c_7x1_Activation")(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name="Block17_1_Concatenate")(branches)
        up = Conv2D(896, 1, strides=1, padding="same", use_bias=True, 
                    name="Block17_1_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.1})(up)
        x = add([x, up])
        x = Activation("relu", name="Block17_1_Activation")(x)
        branch_0 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, 
                        name="Block17_2_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_2_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block17_2_Branch_0_Conv2d_1x1_Activation")(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, 
                        name="Block17_2_Branch_2_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_2_Branch_2_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_2_Branch_2_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding="same", use_bias=False, 
                        name="Block17_2_Branch_2_Conv2d_0b_1x7")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_2_Branch_2_Conv2d_0b_1x7_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_2_Branch_2_Conv2d_0b_1x7_Activation")(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding="same", use_bias=False, 
                        name="Block17_2_Branch_2_Conv2d_0c_7x1")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_2_Branch_2_Conv2d_0c_7x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_2_Branch_2_Conv2d_0c_7x1_Activation")(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name="Block17_2_Concatenate")(branches)
        up = Conv2D(896, 1, strides=1, padding="same", use_bias=True, 
                    name="Block17_2_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.1})(up)
        x = add([x, up])
        x = Activation("relu", name="Block17_2_Activation")(x)
        branch_0 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, 
                        name="Block17_3_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_3_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block17_3_Branch_0_Conv2d_1x1_Activation")(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, 
                        name="Block17_3_Branch_3_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_3_Branch_3_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_3_Branch_3_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding="same", use_bias=False, 
                        name="Block17_3_Branch_3_Conv2d_0b_1x7")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_3_Branch_3_Conv2d_0b_1x7_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_3_Branch_3_Conv2d_0b_1x7_Activation")(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding="same", use_bias=False, 
                        name="Block17_3_Branch_3_Conv2d_0c_7x1")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_3_Branch_3_Conv2d_0c_7x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_3_Branch_3_Conv2d_0c_7x1_Activation")(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name="Block17_3_Concatenate")(branches)
        up = Conv2D(896, 1, strides=1, padding="same", use_bias=True, 
                    name="Block17_3_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.1})(up)
        x = add([x, up])
        x = Activation("relu", name="Block17_3_Activation")(x)
        branch_0 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, 
                        name="Block17_4_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_4_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block17_4_Branch_0_Conv2d_1x1_Activation")(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, 
                        name="Block17_4_Branch_4_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_4_Branch_4_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_4_Branch_4_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding="same", use_bias=False, 
                        name="Block17_4_Branch_4_Conv2d_0b_1x7")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_4_Branch_4_Conv2d_0b_1x7_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_4_Branch_4_Conv2d_0b_1x7_Activation")(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding="same", use_bias=False, 
                        name="Block17_4_Branch_4_Conv2d_0c_7x1")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_4_Branch_4_Conv2d_0c_7x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_4_Branch_4_Conv2d_0c_7x1_Activation")(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name="Block17_4_Concatenate")(branches)
        up = Conv2D(896, 1, strides=1, padding="same", use_bias=True, 
                    name="Block17_4_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.1})(up)
        x = add([x, up])
        x = Activation("relu", name="Block17_4_Activation")(x)
        branch_0 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, 
                        name="Block17_5_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_5_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block17_5_Branch_0_Conv2d_1x1_Activation")(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, 
                        name="Block17_5_Branch_5_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_5_Branch_5_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_5_Branch_5_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding="same", use_bias=False, 
                        name="Block17_5_Branch_5_Conv2d_0b_1x7")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_5_Branch_5_Conv2d_0b_1x7_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_5_Branch_5_Conv2d_0b_1x7_Activation")(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding="same", use_bias=False, 
                        name="Block17_5_Branch_5_Conv2d_0c_7x1")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_5_Branch_5_Conv2d_0c_7x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_5_Branch_5_Conv2d_0c_7x1_Activation")(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name="Block17_5_Concatenate")(branches)
        up = Conv2D(896, 1, strides=1, padding="same", use_bias=True, 
                    name="Block17_5_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.1})(up)
        x = add([x, up])
        x = Activation("relu", name="Block17_5_Activation")(x)
        branch_0 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, 
                        name="Block17_6_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_6_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block17_6_Branch_0_Conv2d_1x1_Activation")(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, 
                        name="Block17_6_Branch_6_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_6_Branch_6_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_6_Branch_6_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding="same", use_bias=False, 
                        name="Block17_6_Branch_6_Conv2d_0b_1x7")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_6_Branch_6_Conv2d_0b_1x7_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_6_Branch_6_Conv2d_0b_1x7_Activation")(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding="same", use_bias=False, 
                        name="Block17_6_Branch_6_Conv2d_0c_7x1")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_6_Branch_6_Conv2d_0c_7x1_BatchNorm",)(branch_1)
        branch_1 = Activation("relu", name="Block17_6_Branch_6_Conv2d_0c_7x1_Activation")(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name="Block17_6_Concatenate")(branches)
        up = Conv2D(896, 1, strides=1, padding="same", use_bias=True, 
                    name="Block17_6_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.1})(up)
        x = add([x, up])
        x = Activation("relu", name="Block17_6_Activation")(x)
        branch_0 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, 
                        name="Block17_7_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_7_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block17_7_Branch_0_Conv2d_1x1_Activation")(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, 
                        name="Block17_7_Branch_7_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_7_Branch_7_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_7_Branch_7_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding="same", use_bias=False, 
                        name="Block17_7_Branch_7_Conv2d_0b_1x7",)(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_7_Branch_7_Conv2d_0b_1x7_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_7_Branch_7_Conv2d_0b_1x7_Activation")(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding="same", use_bias=False, 
                        name="Block17_7_Branch_7_Conv2d_0c_7x1")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_7_Branch_7_Conv2d_0c_7x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_7_Branch_7_Conv2d_0c_7x1_Activation")(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name="Block17_7_Concatenate")(branches)
        up = Conv2D(896, 1, strides=1, padding="same", use_bias=True, 
                    name="Block17_7_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.1})(up)
        x = add([x, up])
        x = Activation("relu", name="Block17_7_Activation")(x)
        branch_0 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, 
                        name="Block17_8_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_8_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block17_8_Branch_0_Conv2d_1x1_Activation")(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, 
                        name="Block17_8_Branch_8_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_8_Branch_8_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_8_Branch_8_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding="same", use_bias=False, 
                        name="Block17_8_Branch_8_Conv2d_0b_1x7")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_8_Branch_8_Conv2d_0b_1x7_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_8_Branch_8_Conv2d_0b_1x7_Activation")(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding="same", use_bias=False, 
                        name="Block17_8_Branch_8_Conv2d_0c_7x1")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_8_Branch_8_Conv2d_0c_7x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_8_Branch_8_Conv2d_0c_7x1_Activation")(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name="Block17_8_Concatenate")(branches)
        up = Conv2D(896, 1, strides=1, padding="same", use_bias=True, 
                    name="Block17_8_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.1})(up)
        x = add([x, up])
        x = Activation("relu", name="Block17_8_Activation")(x)
        branch_0 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, 
                        name="Block17_9_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_9_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block17_9_Branch_0_Conv2d_1x1_Activation")(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, 
                        name="Block17_9_Branch_9_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_9_Branch_9_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_9_Branch_9_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding="same", use_bias=False, 
                        name="Block17_9_Branch_9_Conv2d_0b_1x7")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_9_Branch_9_Conv2d_0b_1x7_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_9_Branch_9_Conv2d_0b_1x7_Activation")(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding="same", use_bias=False, 
                        name="Block17_9_Branch_9_Conv2d_0c_7x1")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_9_Branch_9_Conv2d_0c_7x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_9_Branch_9_Conv2d_0c_7x1_Activation")(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name="Block17_9_Concatenate")(branches)
        up = Conv2D(896, 1, strides=1, padding="same", use_bias=True, 
                    name="Block17_9_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.1})(up)
        x = add([x, up])
        x = Activation("relu", name="Block17_9_Activation")(x)
        branch_0 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, 
                        name="Block17_10_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_10_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block17_10_Branch_0_Conv2d_1x1_Activation")(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, 
                        name="Block17_10_Branch_10_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_10_Branch_10_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_10_Branch_10_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding="same", use_bias=False, 
                        name="Block17_10_Branch_10_Conv2d_0b_1x7")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_10_Branch_10_Conv2d_0b_1x7_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_10_Branch_10_Conv2d_0b_1x7_Activation")(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding="same", use_bias=False,
                        name="Block17_10_Branch_10_Conv2d_0c_7x1")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block17_10_Branch_10_Conv2d_0c_7x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block17_10_Branch_10_Conv2d_0c_7x1_Activation")(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name="Block17_10_Concatenate")(branches)
        up = Conv2D(896, 1, strides=1, padding="same", use_bias=True, 
                    name="Block17_10_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.1})(up)
        x = add([x, up])
        x = Activation("relu", name="Block17_10_Activation")(x)
        branch_0 = Conv2D(256, 1, strides=1, padding="same", use_bias=False, 
                        name="Mixed_7a_Branch_0_Conv2d_0a_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Mixed_7a_Branch_0_Conv2d_0a_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Mixed_7a_Branch_0_Conv2d_0a_1x1_Activation")(branch_0)
        branch_0 = Conv2D(384, 3, strides=2, padding="valid", use_bias=False, 
                        name="Mixed_7a_Branch_0_Conv2d_1a_3x3")(branch_0)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Mixed_7a_Branch_0_Conv2d_1a_3x3_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Mixed_7a_Branch_0_Conv2d_1a_3x3_Activation")(branch_0)
        branch_1 = Conv2D(256, 1, strides=1, padding="same", use_bias=False, 
                        name="Mixed_7a_Branch_1_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Mixed_7a_Branch_1_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Mixed_7a_Branch_1_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(256, 3, strides=2, padding="valid", use_bias=False, 
                        name="Mixed_7a_Branch_1_Conv2d_1a_3x3")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Mixed_7a_Branch_1_Conv2d_1a_3x3_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Mixed_7a_Branch_1_Conv2d_1a_3x3_Activation")(branch_1)
        branch_2 = Conv2D(256, 1, strides=1, padding="same", use_bias=False, 
                        name="Mixed_7a_Branch_2_Conv2d_0a_1x1")(x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Mixed_7a_Branch_2_Conv2d_0a_1x1_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Mixed_7a_Branch_2_Conv2d_0a_1x1_Activation")(branch_2)
        branch_2 = Conv2D(256, 3, strides=1, padding="same", use_bias=False, 
                        name="Mixed_7a_Branch_2_Conv2d_0b_3x3")(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Mixed_7a_Branch_2_Conv2d_0b_3x3_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Mixed_7a_Branch_2_Conv2d_0b_3x3_Activation")(branch_2)
        branch_2 = Conv2D(256, 3, strides=2, padding="valid", use_bias=False, 
                        name="Mixed_7a_Branch_2_Conv2d_1a_3x3")(branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Mixed_7a_Branch_2_Conv2d_1a_3x3_BatchNorm")(branch_2)
        branch_2 = Activation("relu", name="Mixed_7a_Branch_2_Conv2d_1a_3x3_Activation")(branch_2)
        branch_pool = MaxPooling2D(3, strides=2, padding="valid", 
                                name="Mixed_7a_Branch_3_MaxPool_1a_3x3")(x)
        branches = [branch_0, branch_1, branch_2, branch_pool]
        x = Concatenate(axis=3, name="Mixed_7a")(branches)
        branch_0 = Conv2D(192, 1, strides=1, padding="same", use_bias=False, 
                        name="Block8_1_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block8_1_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block8_1_Branch_0_Conv2d_1x1_Activation")(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding="same", use_bias=False, 
                        name="Block8_1_Branch_1_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block8_1_Branch_1_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_1_Branch_1_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding="same", use_bias=False, 
                        name="Block8_1_Branch_1_Conv2d_0b_1x3")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block8_1_Branch_1_Conv2d_0b_1x3_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_1_Branch_1_Conv2d_0b_1x3_Activation")(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding="same", use_bias=False, 
                        name="Block8_1_Branch_1_Conv2d_0c_3x1")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block8_1_Branch_1_Conv2d_0c_3x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_1_Branch_1_Conv2d_0c_3x1_Activation")(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name="Block8_1_Concatenate")(branches)
        up = Conv2D(1792, 1, strides=1, padding="same", use_bias=True, 
                    name="Block8_1_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.2})(up)
        x = add([x, up])
        x = Activation("relu", name="Block8_1_Activation")(x)
        branch_0 = Conv2D(192, 1, strides=1, padding="same", use_bias=False, 
                        name="Block8_2_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block8_2_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block8_2_Branch_0_Conv2d_1x1_Activation")(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding="same", use_bias=False, 
                        name="Block8_2_Branch_2_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block8_2_Branch_2_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_2_Branch_2_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding="same", use_bias=False, 
                        name="Block8_2_Branch_2_Conv2d_0b_1x3")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block8_2_Branch_2_Conv2d_0b_1x3_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_2_Branch_2_Conv2d_0b_1x3_Activation")(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding="same", use_bias=False, 
                        name="Block8_2_Branch_2_Conv2d_0c_3x1")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block8_2_Branch_2_Conv2d_0c_3x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_2_Branch_2_Conv2d_0c_3x1_Activation")(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name="Block8_2_Concatenate")(branches)
        up = Conv2D(1792, 1, strides=1, padding="same", use_bias=True, 
                    name="Block8_2_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.2})(up)
        x = add([x, up])
        x = Activation("relu", name="Block8_2_Activation")(x)
        branch_0 = Conv2D(192, 1, strides=1, padding="same", use_bias=False, 
                        name="Block8_3_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block8_3_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block8_3_Branch_0_Conv2d_1x1_Activation")(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding="same", use_bias=False, 
                        name="Block8_3_Branch_3_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block8_3_Branch_3_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_3_Branch_3_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding="same", use_bias=False, 
                        name="Block8_3_Branch_3_Conv2d_0b_1x3")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block8_3_Branch_3_Conv2d_0b_1x3_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_3_Branch_3_Conv2d_0b_1x3_Activation")(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding="same", use_bias=False, 
                        name="Block8_3_Branch_3_Conv2d_0c_3x1")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block8_3_Branch_3_Conv2d_0c_3x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_3_Branch_3_Conv2d_0c_3x1_Activation")(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name="Block8_3_Concatenate")(branches)
        up = Conv2D(1792, 1, strides=1, padding="same", use_bias=True, 
                    name="Block8_3_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.2})(up)
        x = add([x, up])
        x = Activation("relu", name="Block8_3_Activation")(x)
        branch_0 = Conv2D(192, 1, strides=1, padding="same", use_bias=False, 
                        name="Block8_4_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block8_4_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block8_4_Branch_0_Conv2d_1x1_Activation")(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding="same", use_bias=False, 
                        name="Block8_4_Branch_4_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block8_4_Branch_4_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_4_Branch_4_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding="same", use_bias=False, 
                        name="Block8_4_Branch_4_Conv2d_0b_1x3",)(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block8_4_Branch_4_Conv2d_0b_1x3_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_4_Branch_4_Conv2d_0b_1x3_Activation")(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding="same", use_bias=False, 
                        name="Block8_4_Branch_4_Conv2d_0c_3x1")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block8_4_Branch_4_Conv2d_0c_3x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_4_Branch_4_Conv2d_0c_3x1_Activation")(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name="Block8_4_Concatenate")(branches)
        up = Conv2D(1792, 1, strides=1, padding="same", use_bias=True, 
                    name="Block8_4_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.2})(up)
        x = add([x, up])
        x = Activation("relu", name="Block8_4_Activation")(x)
        branch_0 = Conv2D(192, 1, strides=1, padding="same", use_bias=False, 
                        name="Block8_5_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block8_5_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block8_5_Branch_0_Conv2d_1x1_Activation")(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding="same", use_bias=False, 
                        name="Block8_5_Branch_5_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block8_5_Branch_5_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_5_Branch_5_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding="same", use_bias=False, 
                        name="Block8_5_Branch_5_Conv2d_0b_1x3")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block8_5_Branch_5_Conv2d_0b_1x3_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_5_Branch_5_Conv2d_0b_1x3_Activation")(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding="same", use_bias=False, 
                        name="Block8_5_Branch_5_Conv2d_0c_3x1")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block8_5_Branch_5_Conv2d_0c_3x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_5_Branch_5_Conv2d_0c_3x1_Activation")(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name="Block8_5_Concatenate")(branches)
        up = Conv2D(1792, 1, strides=1, padding="same", use_bias=True, 
                    name="Block8_5_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 0.2})(up)
        x = add([x, up])
        x = Activation("relu", name="Block8_5_Activation")(x)
        branch_0 = Conv2D(192, 1, strides=1, padding="same", use_bias=False, 
                        name="Block8_6_Branch_0_Conv2d_1x1")(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block8_6_Branch_0_Conv2d_1x1_BatchNorm")(branch_0)
        branch_0 = Activation("relu", name="Block8_6_Branch_0_Conv2d_1x1_Activation")(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding="same", use_bias=False, 
                        name="Block8_6_Branch_1_Conv2d_0a_1x1")(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block8_6_Branch_1_Conv2d_0a_1x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_6_Branch_1_Conv2d_0a_1x1_Activation")(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding="same", use_bias=False, 
                        name="Block8_6_Branch_1_Conv2d_0b_1x3")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block8_6_Branch_1_Conv2d_0b_1x3_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_6_Branch_1_Conv2d_0b_1x3_Activation")(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding="same", use_bias=False, 
                        name="Block8_6_Branch_1_Conv2d_0c_3x1")(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, 
                                    name="Block8_6_Branch_1_Conv2d_0c_3x1_BatchNorm")(branch_1)
        branch_1 = Activation("relu", name="Block8_6_Branch_1_Conv2d_0c_3x1_Activation")(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name="Block8_6_Concatenate")(branches)
        up = Conv2D(1792, 1, strides=1, padding="same", use_bias=True, name="Block8_6_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=int_shape(up)[1:], arguments={"scale": 1})(up)
        x = add([x, up])
        x = GlobalAveragePooling2D(name="AvgPool")(x)
        x = Dropout(1.0 - 0.8, name="Dropout")(x)
        x = Dense(dimension, use_bias=False, name="Bottleneck")(x)
        x = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False, name="Bottleneck_BatchNorm")(x)
        return Model(inputs, x, name="inception_resnet_v1")

    def load_model(self, 
                   url: str = DOWNLOAD_URL + "facenet_weights.h5") -> Model:
        """Construct FaceNet model (128d or 512d), download its weights and load
        Args:
            dimension (int): construct FaceNet-128d or FaceNet-512d models
            url (str): URL to the weight file
        Returns:
            model (Model)"""
        if self.__class__.__name__ == "FaceNet128dClient":
            model = FaceNetBase._inception_res_netV2()
            output = get_deepface_home() + "/.deepface/weights/facenet_weights.h5"
        else:
            model = FaceNetBase._inception_res_netV2(512)
            url = DOWNLOAD_URL + "facenet512_weights.h5"
            output = get_deepface_home() + "/.deepface/weights/facenet512_weights.h5"
        self._download(url, output)
        model.load_weights(output)
        return model
