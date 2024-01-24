from typing import List
from os.path import isfile

import tensorflow as tf
from zipfile import ZipFile
from bz2 import BZ2File
from dlib import face_recognition_model_v1
from gdown import download
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import (Activation, ZeroPadding2D, Input, 
    Conv2D, BatchNormalization, MaxPooling2D, PReLU, Add, Dropout, Flatten, Dense,
    Concatenate, Lambda, GlobalAveragePooling2D, AveragePooling2D, LocallyConnected2D, 
    concatenate, Convolution2D, add)
import numpy as np

from recognition_base import FacialRecognitionBase
from utils.functions import get_deepface_home


class ArcFaceClient(FacialRecognitionBase):
    """ArcFace model class"""
    def __init__(self) -> None:
        self.model = self.load_model()
        self.model_name = "ArcFace"

    def find_embeddings(self, img: np.ndarray) -> List[float]:
        """
        Find embeddings with ArcFace model
        Args:
            img (np.ndarray): pre-loaded image in BGR
        Returns
            embeddings (list): multi-dimensional vector
        """
        return self.model(img, training=False).numpy()[0].tolist()

    @staticmethod
    def load_model(
        url="https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5",
    ) -> Model:
        """Construct ArcFace model, download its weights and load
        Returns:
            model (Model)"""
        base_model = ArcFaceClient.ResNet34()
        inputs = base_model.inputs[0]
        arcface_model = base_model.outputs[0]
        arcface_model = BatchNormalization(momentum=0.9, epsilon=2e-5)(arcface_model)
        arcface_model = Dropout(0.4)(arcface_model)
        arcface_model = Flatten()(arcface_model)
        arcface_model = Dense(512, activation=None, use_bias=True, 
                              kernel_initializer="glorot_normal")(arcface_model)
        embedding = BatchNormalization(momentum=0.9, epsilon=2e-5, 
                                       name="embedding", scale=True)(arcface_model)
        model = Model(inputs, embedding, name=base_model.name)
        output = get_deepface_home() + "/.deepface/weights/" + "arcface_weights.h5"

        if isfile(output) != True:
            download(url, output, quiet=False)
        
        model.load_weights(output)
        return model

    @staticmethod
    def ResNet34() -> Model:
        """
        ResNet34 model
        Returns:
            model (Model)
        """
        img_input = Input(shape=(112, 112, 3))

        x = ZeroPadding2D(padding=1, name="conv1_pad")(img_input)
        x = Conv2D(64, 3, strides=1, use_bias=False, 
                   kernel_initializer="glorot_normal", name="conv1_conv")(x)
        x = BatchNormalization(axis=3, epsilon=2e-5, momentum=0.9, name="conv1_bn")(x)
        x = PReLU(shared_axes=[1, 2], name="conv1_prelu")(x)
        x = ArcFaceClient.stack_fn(x)

        return training.Model(img_input, x, name="ResNet34")

    @staticmethod
    def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
        bn_axis = 3

        if conv_shortcut:
            shortcut = Conv2D(filters,
                              1,
                              strides=stride,
                              use_bias=False,
                              kernel_initializer="glorot_normal",
                              name=name + "_0_conv")(x)
            shortcut = BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, 
                                          name=name + "_0_bn")(shortcut)
        else:
            shortcut = x

        x = BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + "_1_bn")(x)
        x = ZeroPadding2D(padding=1, name=name + "_1_pad")(x)
        x = Conv2D(filters,
                   3,
                   strides=1,
                   kernel_initializer="glorot_normal",
                   use_bias=False,
                   name=name + "_1_conv")(x)
        x = BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + "_2_bn")(x)
        x = PReLU(shared_axes=[1, 2], name=name + "_1_prelu")(x)
        x = ZeroPadding2D(padding=1, name=name + "_2_pad")(x)
        x = Conv2D(filters,
                   kernel_size,
                   strides=stride,
                   kernel_initializer="glorot_normal",
                   use_bias=False,
                   name=name + "_2_conv")(x)
        x = BatchNormalization(axis=bn_axis, epsilon=2e-5, momentum=0.9, name=name + "_3_bn")(x)

        return Add(name=name + "_add")([shortcut, x])

    @staticmethod
    def stack1(x, filters, blocks, stride1=2, name=None):
        x = ArcFaceClient.block1(x, filters, stride=stride1, name=name + "_block1")
        for i in range(2, blocks + 1):
            x = ArcFaceClient.block1(x, filters, conv_shortcut=False, name=name + "_block" + str(i))
        return x

    @staticmethod
    def stack_fn(x):
        x = ArcFaceClient.stack1(x, 64, 3, name="conv2")
        x = ArcFaceClient.stack1(x, 128, 4, name="conv3")
        x = ArcFaceClient.stack1(x, 256, 6, name="conv4")
        return ArcFaceClient.stack1(x, 512, 3, name="conv5")


class DeepIdClient(FacialRecognitionBase):
    """DeepId model class"""
    def __init__(self) -> None:
        self.model = self.load_model()
        self.model_name = "DeepId"

    def find_embeddings(self, img: np.ndarray) -> List[float]:
        """Find embeddings with DeepId model
        Args:
            img (np.ndarray): pre-loaded image in BGR
        Returns
            embeddings (list): multi-dimensional vector"""
        return self.model(img, training=False).numpy()[0].tolist()

    @staticmethod
    def load_model(
        url="https://github.com/serengil/deepface_models/releases/download/v1.0/deepid_keras_weights.h5",
    ) -> Model:
        """Construct DeepId model, download its weights and load"""
        myInput = Input(shape=(55, 47, 3))

        x = Conv2D(20, (4, 4), name="Conv1", activation="relu", input_shape=(55, 47, 3))(myInput)
        x = MaxPooling2D(pool_size=2, strides=2, name="Pool1")(x)
        x = Dropout(rate=0.99, name="D1")(x)

        x = Conv2D(40, (3, 3), name="Conv2", activation="relu")(x)
        x = MaxPooling2D(pool_size=2, strides=2, name="Pool2")(x)
        x = Dropout(rate=0.99, name="D2")(x)

        x = Conv2D(60, (3, 3), name="Conv3", activation="relu")(x)
        x = MaxPooling2D(pool_size=2, strides=2, name="Pool3")(x)
        x = Dropout(rate=0.99, name="D3")(x)
        
        x1 = Flatten()(x)
        fc11 = Dense(160, name="fc11")(x1)
        
        x2 = Conv2D(80, (2, 2), name="Conv4", activation="relu")(x)
        x2 = Flatten()(x2)
        fc12 = Dense(160, name="fc12")(x2)
        
        y = Add()([fc11, fc12])
        y = Activation("relu", name="deepid")(y)

        model = Model(inputs=[myInput], outputs=y)
        home = get_deepface_home()

        if isfile(home + "/.deepface/weights/deepid_keras_weights.h5") != True:
            output = home + "/.deepface/weights/deepid_keras_weights.h5"
            download(url, output, quiet=False)

        model.load_weights(home + "/.deepface/weights/deepid_keras_weights.h5")
        return model


class DlibResNet:
    def __init__(self):
        self.layers = [DlibMetaData()]

        home = get_deepface_home()
        weight_file = home + "/.deepface/weights/dlib_face_recognition_resnet_model_v1.dat"

        if isfile(weight_file) != True:
            file_name = "dlib_face_recognition_resnet_model_v1.dat.bz2"
            output = f"{home}/.deepface/weights/{file_name}"
            download(f"http://dlib.net/files/{file_name}", output, quiet=False)

            zipfile = BZ2File(output)
            data = zipfile.read()
            newfilepath = output[:-4]
            with open(newfilepath, "wb") as f:
                f.write(data)

        self.model = face_recognition_model_v1(weight_file)


class DlibMetaData:
    def __init__(self):
        self.input_shape = [[1, 150, 150, 3]]


class DlibClient(FacialRecognitionBase):
    """Dlib model class"""
    def __init__(self):
        self.model = DlibResNet()
        self.model_name = "Dlib"

    def find_embeddings(self, img: np.ndarray) -> List[float]:
        """Find embeddings with Dlib model - different than regular models
        Args:
            img (np.ndarray): pre-loaded image in BGR
        Returns
            embeddings (list): multi-dimensional vector"""
        if len(img.shape) == 4:
            img = img[0]

        img = img[:, :, ::-1]

        if img.max() <= 1:
            img = img * 255

        img = img.astype(np.uint8)

        img_representation = self.model.model.compute_face_descriptor(img)
        img_representation = np.array(img_representation)
        return np.expand_dims(img_representation, axis=0)[0].tolist()


class FaceNetBase(FacialRecognitionBase):
    @staticmethod
    def _scaling(x, scale):
        return x * scale

    @staticmethod
    def _inception_res_netV2(dimension: int = 128) -> Model:
        """InceptionResNetV2 model
        Args:
            dimension (int): number of dimensions in the embedding layer
        Returns:
            model (Model)"""
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
        up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.17})(up)
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
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block35_2_Branch_2_Conv2d_0b_3x3_BatchNorm",
        )(branch_2)
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
        up = Lambda(FaceNetBase._scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.17})(up)
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
        up = Lambda(FaceNetBase._scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.17})(up)
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
        up = Lambda(FaceNetBase._scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.17})(up)
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
        up = Lambda(FaceNetBase._scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.17})(up)
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
        up = Lambda(FaceNetBase._scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.1})(up)
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
        up = Lambda(FaceNetBase._scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.1})(up)
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
        up = Lambda(FaceNetBase._scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.1})(up)
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
        up = Lambda(FaceNetBase._scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.1})(up)
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
        up = Lambda(FaceNetBase._scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.1})(up)
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
        up = Lambda(FaceNetBase._scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.1})(up)
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
        up = Lambda(FaceNetBase._scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.1})(up)
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
        up = Lambda(FaceNetBase._scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.1})(up)
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
        up = Lambda(FaceNetBase._scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.1})(up)
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
        up = Lambda(FaceNetBase._scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.1})(up)
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
        up = Lambda(FaceNetBase._scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.2})(up)
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
        up = Lambda(FaceNetBase._scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.2})(up)
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
        up = Lambda(FaceNetBase._scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.2})(up)
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
        up = Lambda(FaceNetBase._scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.2})(up)
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
        up = Lambda(FaceNetBase._scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.2})(up)
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
        up = Conv2D(1792, 1, strides=1, padding="same", use_bias=True, 
                    name="Block8_6_Conv2d_1x1")(mixed)
        up = Lambda(FaceNetBase._scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 1})(up)
        x = add([x, up])
        x = GlobalAveragePooling2D(name="AvgPool")(x)
        x = Dropout(1.0 - 0.8, name="Dropout")(x)
        x = Dense(dimension, use_bias=False, name="Bottleneck")(x)
        x = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False, 
                            name="Bottleneck_BatchNorm")(x)

        return Model(inputs, x, name="inception_resnet_v1")

    @staticmethod
    def load_facenet128d_model(
        url="https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.h5"
    ) -> Model:
        """Construct FaceNet-128d model, download weights and then load weights
        Args:
            dimension (int): construct FaceNet-128d or FaceNet-512d models
        Returns:
            model (Model)"""
        model = FaceNetBase._inception_res_netV2()
        output = get_deepface_home() + "/.deepface/weights/facenet_weights.h5"
        if isfile(output) != True:
            download(url, output, quiet=False)

        model.load_weights(output)

        return model

    @staticmethod
    def load_facenet512d_model(
        url="https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5"
    ) -> Model:
        """Construct FaceNet-512d model, download its weights and load
        Returns:
            model (Model)"""
        model = FaceNetBase._inception_res_netV2(dimension=512)
        output = get_deepface_home() + "/.deepface/weights/facenet512_weights.h5"
        if isfile(output) != True:
            download(url, output, quiet=False)

        model.load_weights(output)

        return model


class FaceNet128dClient(FaceNetBase):
    """FaceNet-128d model class"""
    def __init__(self):
        self.model = super().load_facenet128d_model()
        self.model_name = "FaceNet-128d"

    def find_embeddings(self, img: np.ndarray) -> List[float]:
        """Find embeddings with FaceNet-128d model
        Args:
            img (np.ndarray): pre-loaded image in BGR
        Returns
            embeddings (list): multi-dimensional vector"""
        return self.model(img, training=False).numpy()[0].tolist()


class FaceNet512dClient(FacialRecognitionBase):
    """FaceNet-512d model class"""
    def __init__(self):
        self.model = super().load_facenet512d_model()
        self.model_name = "FaceNet-512d"

    def find_embeddings(self, img: np.ndarray) -> List[float]:
        """Find embeddings with FaceNet-512d model
        Args:
            img (np.ndarray): pre-loaded image in BGR
        Returns
            embeddings (list): multi-dimensional vector"""
        return self.model(img, training=False).numpy()[0].tolist()


class DeepFaceClient(FacialRecognitionBase):
    """Fb's DeepFace model class"""
    def __init__(self):
        self.model = self.load_model()
        self.model_name = "DeepFace"

    def find_embeddings(self, img: np.ndarray) -> List[float]:
        """Find embeddings with OpenFace model
        Args:
            img (np.ndarray): pre-loaded image in BGR
        Returns
            embeddings (list): multi-dimensional vector"""
        return self.model(img, training=False).numpy()[0].tolist()

    @staticmethod
    def load_model(
        url="https://github.com/swghosh/DeepFace/releases/download/weights-vggface2-2d-aligned/VGGFace2_DeepFace_weights_val-0.9034.h5.zip",
    ) -> Model:
        """Construct DeepFace model, download its weights and load"""
        base_model = Sequential()
        base_model.add(
            Convolution2D(32, (11, 11), activation="relu", name="C1", input_shape=(152, 152, 3))
        )
        base_model.add(MaxPooling2D(pool_size=3, strides=2, padding="same", name="M2"))
        base_model.add(Convolution2D(16, (9, 9), activation="relu", name="C3"))
        base_model.add(LocallyConnected2D(16, (9, 9), activation="relu", name="L4"))
        base_model.add(LocallyConnected2D(16, (7, 7), strides=2, activation="relu", name="L5"))
        base_model.add(LocallyConnected2D(16, (5, 5), activation="relu", name="L6"))
        base_model.add(Flatten(name="F0"))
        base_model.add(Dense(4096, activation="relu", name="F7"))
        base_model.add(Dropout(rate=0.5, name="D0"))
        base_model.add(Dense(8631, activation="softmax", name="F8"))

        home = get_deepface_home()

        dr = home + "/.deepface/weights/VGGFace2_DeepFace_weights_val-0.9034.h5"
        if isfile(dr) != True:
            output = dr + ".zip"
            download(url, output, quiet=False)

            with ZipFile(output, "r") as zip_ref:
                zip_ref.extractall(home + "/.deepface/weights/")

        base_model.load_weights(dr)

        return Model(inputs=base_model.layers[0].input, outputs=base_model.layers[-3].output)


class OpenFaceClient(FacialRecognitionBase):
    """OpenFace model class"""
    def __init__(self):
        self.model = self.load_model()
        self.model_name = "OpenFace"

    def find_embeddings(self, img: np.ndarray) -> List[float]:
        """Find embeddings with OpenFace model
        Args:
            img (np.ndarray): pre-loaded image in BGR
        Returns
            embeddings (list): multi-dimensional vector"""
        return self.model(img, training=False).numpy()[0].tolist()

    @staticmethod
    def load_model(
        url="https://github.com/serengil/deepface_models/releases/download/v1.0/openface_weights.h5",
    ) -> Model:
        """
        Consturct OpenFace model, download its weights and load
        Returns:
            model (Model)
        """
        myInput = Input(shape=(96, 96, 3))
        x = ZeroPadding2D(padding=(3, 3), input_shape=(96, 96, 3))(myInput)
        x = Conv2D(64, (7, 7), strides=(2, 2), name="conv1")(x)
        x = BatchNormalization(axis=3, epsilon=0.00001, name="bn1")(x)
        x = Activation("relu")(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = MaxPooling2D(pool_size=3, strides=2)(x)
        x = Lambda(lambda x: tf.nn.lrn(x, alpha=1e-4, beta=0.75), name="lrn_1")(x)
        x = Conv2D(64, (1, 1), name="conv2")(x)
        x = BatchNormalization(axis=3, epsilon=0.00001, name="bn2")(x)
        x = Activation("relu")(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(192, (3, 3), name="conv3")(x)
        x = BatchNormalization(axis=3, epsilon=0.00001, name="bn3")(x)
        x = Activation("relu")(x)
        x = Lambda(lambda x: tf.nn.lrn(x, alpha=1e-4, beta=0.75), name="lrn_2")(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = MaxPooling2D(pool_size=3, strides=2)(x)
        inception_3a_3x3 = Conv2D(96, (1, 1), name="inception_3a_3x3_conv1")(x)
        inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, 
                                              name="inception_3a_3x3_bn1")(inception_3a_3x3)
        inception_3a_3x3 = Activation("relu")(inception_3a_3x3)
        inception_3a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3a_3x3)
        inception_3a_3x3 = Conv2D(128, (3, 3), name="inception_3a_3x3_conv2")(inception_3a_3x3)
        inception_3a_3x3 = BatchNormalization(axis=3, epsilon=0.00001, 
                                              name="inception_3a_3x3_bn2")(inception_3a_3x3)
        inception_3a_3x3 = Activation("relu")(inception_3a_3x3)
        inception_3a_5x5 = Conv2D(16, (1, 1), name="inception_3a_5x5_conv1")(x)
        inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, 
                                              name="inception_3a_5x5_bn1")(inception_3a_5x5)
        inception_3a_5x5 = Activation("relu")(inception_3a_5x5)
        inception_3a_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3a_5x5)
        inception_3a_5x5 = Conv2D(32, (5, 5), name="inception_3a_5x5_conv2")(inception_3a_5x5)
        inception_3a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, 
                                              name="inception_3a_5x5_bn2")(inception_3a_5x5)
        inception_3a_5x5 = Activation("relu")(inception_3a_5x5)
        inception_3a_pool = MaxPooling2D(pool_size=3, strides=2)(x)
        inception_3a_pool = Conv2D(32, (1, 1), name="inception_3a_pool_conv")(inception_3a_pool)
        inception_3a_pool = BatchNormalization(axis=3, epsilon=0.00001, 
                                               name="inception_3a_pool_bn")(inception_3a_pool)
        inception_3a_pool = Activation("relu")(inception_3a_pool)
        inception_3a_pool = ZeroPadding2D(padding=((3, 4), (3, 4)))(inception_3a_pool)
        inception_3a_1x1 = Conv2D(64, (1, 1), name="inception_3a_1x1_conv")(x)
        inception_3a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, 
                                              name="inception_3a_1x1_bn")(inception_3a_1x1)
        inception_3a_1x1 = Activation("relu")(inception_3a_1x1)
        inception_3a = concatenate([inception_3a_3x3, inception_3a_5x5, 
                                    inception_3a_pool, inception_3a_1x1], axis=3)
        inception_3b_3x3 = Conv2D(96, (1, 1), name="inception_3b_3x3_conv1")(inception_3a)
        inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_3b_3x3_bn1")(
            inception_3b_3x3
        )
        inception_3b_3x3 = Activation("relu")(inception_3b_3x3)
        inception_3b_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3b_3x3)
        inception_3b_3x3 = Conv2D(128, (3, 3), name="inception_3b_3x3_conv2")(inception_3b_3x3)
        inception_3b_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_3b_3x3_bn2")(
            inception_3b_3x3
        )
        inception_3b_3x3 = Activation("relu")(inception_3b_3x3)

        inception_3b_5x5 = Conv2D(32, (1, 1), name="inception_3b_5x5_conv1")(inception_3a)
        inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_3b_5x5_bn1")(
            inception_3b_5x5
        )
        inception_3b_5x5 = Activation("relu")(inception_3b_5x5)
        inception_3b_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3b_5x5)
        inception_3b_5x5 = Conv2D(64, (5, 5), name="inception_3b_5x5_conv2")(inception_3b_5x5)
        inception_3b_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_3b_5x5_bn2")(
            inception_3b_5x5
        )
        inception_3b_5x5 = Activation("relu")(inception_3b_5x5)

        inception_3b_pool = Lambda(lambda x: x**2, name="power2_3b")(inception_3a)
        inception_3b_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_3b_pool)
        inception_3b_pool = Lambda(lambda x: x * 9, name="mult9_3b")(inception_3b_pool)
        inception_3b_pool = Lambda(lambda x: K.sqrt(x), name="sqrt_3b")(inception_3b_pool)
        inception_3b_pool = Conv2D(64, (1, 1), name="inception_3b_pool_conv")(inception_3b_pool)
        inception_3b_pool = BatchNormalization(axis=3, epsilon=0.00001, name="inception_3b_pool_bn")(
            inception_3b_pool
        )
        inception_3b_pool = Activation("relu")(inception_3b_pool)
        inception_3b_pool = ZeroPadding2D(padding=(4, 4))(inception_3b_pool)

        inception_3b_1x1 = Conv2D(64, (1, 1), name="inception_3b_1x1_conv")(inception_3a)
        inception_3b_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_3b_1x1_bn")(
            inception_3b_1x1
        )
        inception_3b_1x1 = Activation("relu")(inception_3b_1x1)

        inception_3b = concatenate(
            [inception_3b_3x3, inception_3b_5x5, inception_3b_pool, inception_3b_1x1], axis=3
        )

        # Inception3c
        inception_3c_3x3 = Conv2D(128, (1, 1), strides=(1, 1), name="inception_3c_3x3_conv1")(
            inception_3b
        )
        inception_3c_3x3 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_3c_3x3_bn1")(
            inception_3c_3x3
        )
        inception_3c_3x3 = Activation("relu")(inception_3c_3x3)
        inception_3c_3x3 = ZeroPadding2D(padding=(1, 1))(inception_3c_3x3)
        inception_3c_3x3 = Conv2D(256, (3, 3), strides=(2, 2), name="inception_3c_3x3_conv" + "2")(
            inception_3c_3x3
        )
        inception_3c_3x3 = BatchNormalization(
            axis=3, epsilon=0.00001, name="inception_3c_3x3_bn" + "2"
        )(inception_3c_3x3)
        inception_3c_3x3 = Activation("relu")(inception_3c_3x3)

        inception_3c_5x5 = Conv2D(32, (1, 1), strides=(1, 1), name="inception_3c_5x5_conv1")(
            inception_3b
        )
        inception_3c_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_3c_5x5_bn1")(
            inception_3c_5x5
        )
        inception_3c_5x5 = Activation("relu")(inception_3c_5x5)
        inception_3c_5x5 = ZeroPadding2D(padding=(2, 2))(inception_3c_5x5)
        inception_3c_5x5 = Conv2D(64, (5, 5), strides=(2, 2), name="inception_3c_5x5_conv" + "2")(
            inception_3c_5x5
        )
        inception_3c_5x5 = BatchNormalization(
            axis=3, epsilon=0.00001, name="inception_3c_5x5_bn" + "2"
        )(inception_3c_5x5)
        inception_3c_5x5 = Activation("relu")(inception_3c_5x5)

        inception_3c_pool = MaxPooling2D(pool_size=3, strides=2)(inception_3b)
        inception_3c_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_3c_pool)

        inception_3c = concatenate([inception_3c_3x3, inception_3c_5x5, inception_3c_pool], axis=3)

        # inception 4a
        inception_4a_3x3 = Conv2D(96, (1, 1), strides=(1, 1), name="inception_4a_3x3_conv" + "1")(
            inception_3c
        )
        inception_4a_3x3 = BatchNormalization(
            axis=3, epsilon=0.00001, name="inception_4a_3x3_bn" + "1"
        )(inception_4a_3x3)
        inception_4a_3x3 = Activation("relu")(inception_4a_3x3)
        inception_4a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_4a_3x3)
        inception_4a_3x3 = Conv2D(192, (3, 3), strides=(1, 1), name="inception_4a_3x3_conv" + "2")(
            inception_4a_3x3
        )
        inception_4a_3x3 = BatchNormalization(
            axis=3, epsilon=0.00001, name="inception_4a_3x3_bn" + "2"
        )(inception_4a_3x3)
        inception_4a_3x3 = Activation("relu")(inception_4a_3x3)

        inception_4a_5x5 = Conv2D(32, (1, 1), strides=(1, 1), name="inception_4a_5x5_conv1")(
            inception_3c
        )
        inception_4a_5x5 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_4a_5x5_bn1")(
            inception_4a_5x5
        )
        inception_4a_5x5 = Activation("relu")(inception_4a_5x5)
        inception_4a_5x5 = ZeroPadding2D(padding=(2, 2))(inception_4a_5x5)
        inception_4a_5x5 = Conv2D(64, (5, 5), strides=(1, 1), name="inception_4a_5x5_conv" + "2")(
            inception_4a_5x5
        )
        inception_4a_5x5 = BatchNormalization(
            axis=3, epsilon=0.00001, name="inception_4a_5x5_bn" + "2"
        )(inception_4a_5x5)
        inception_4a_5x5 = Activation("relu")(inception_4a_5x5)

        inception_4a_pool = Lambda(lambda x: x**2, name="power2_4a")(inception_3c)
        inception_4a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_4a_pool)
        inception_4a_pool = Lambda(lambda x: x * 9, name="mult9_4a")(inception_4a_pool)
        inception_4a_pool = Lambda(lambda x: K.sqrt(x), name="sqrt_4a")(inception_4a_pool)

        inception_4a_pool = Conv2D(128, (1, 1), strides=(1, 1), name="inception_4a_pool_conv" + "")(
            inception_4a_pool
        )
        inception_4a_pool = BatchNormalization(
            axis=3, epsilon=0.00001, name="inception_4a_pool_bn" + ""
        )(inception_4a_pool)
        inception_4a_pool = Activation("relu")(inception_4a_pool)
        inception_4a_pool = ZeroPadding2D(padding=(2, 2))(inception_4a_pool)

        inception_4a_1x1 = Conv2D(256, (1, 1), strides=(1, 1), name="inception_4a_1x1_conv" + "")(
            inception_3c
        )
        inception_4a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_4a_1x1_bn" + "")(
            inception_4a_1x1
        )
        inception_4a_1x1 = Activation("relu")(inception_4a_1x1)

        inception_4a = concatenate(
            [inception_4a_3x3, inception_4a_5x5, inception_4a_pool, inception_4a_1x1], axis=3
        )

        # inception4e
        inception_4e_3x3 = Conv2D(160, (1, 1), strides=(1, 1), name="inception_4e_3x3_conv" + "1")(
            inception_4a
        )
        inception_4e_3x3 = BatchNormalization(
            axis=3, epsilon=0.00001, name="inception_4e_3x3_bn" + "1"
        )(inception_4e_3x3)
        inception_4e_3x3 = Activation("relu")(inception_4e_3x3)
        inception_4e_3x3 = ZeroPadding2D(padding=(1, 1))(inception_4e_3x3)
        inception_4e_3x3 = Conv2D(256, (3, 3), strides=(2, 2), name="inception_4e_3x3_conv" + "2")(
            inception_4e_3x3
        )
        inception_4e_3x3 = BatchNormalization(
            axis=3, epsilon=0.00001, name="inception_4e_3x3_bn" + "2"
        )(inception_4e_3x3)
        inception_4e_3x3 = Activation("relu")(inception_4e_3x3)

        inception_4e_5x5 = Conv2D(64, (1, 1), strides=(1, 1), name="inception_4e_5x5_conv" + "1")(
            inception_4a
        )
        inception_4e_5x5 = BatchNormalization(
            axis=3, epsilon=0.00001, name="inception_4e_5x5_bn" + "1"
        )(inception_4e_5x5)
        inception_4e_5x5 = Activation("relu")(inception_4e_5x5)
        inception_4e_5x5 = ZeroPadding2D(padding=(2, 2))(inception_4e_5x5)
        inception_4e_5x5 = Conv2D(128, (5, 5), strides=(2, 2), name="inception_4e_5x5_conv" + "2")(
            inception_4e_5x5
        )
        inception_4e_5x5 = BatchNormalization(
            axis=3, epsilon=0.00001, name="inception_4e_5x5_bn" + "2"
        )(inception_4e_5x5)
        inception_4e_5x5 = Activation("relu")(inception_4e_5x5)

        inception_4e_pool = MaxPooling2D(pool_size=3, strides=2)(inception_4a)
        inception_4e_pool = ZeroPadding2D(padding=((0, 1), (0, 1)))(inception_4e_pool)

        inception_4e = concatenate([inception_4e_3x3, inception_4e_5x5, inception_4e_pool], axis=3)

        # inception5a
        inception_5a_3x3 = Conv2D(96, (1, 1), strides=(1, 1), name="inception_5a_3x3_conv" + "1")(
            inception_4e
        )
        inception_5a_3x3 = BatchNormalization(
            axis=3, epsilon=0.00001, name="inception_5a_3x3_bn" + "1"
        )(inception_5a_3x3)
        inception_5a_3x3 = Activation("relu")(inception_5a_3x3)
        inception_5a_3x3 = ZeroPadding2D(padding=(1, 1))(inception_5a_3x3)
        inception_5a_3x3 = Conv2D(384, (3, 3), strides=(1, 1), name="inception_5a_3x3_conv" + "2")(
            inception_5a_3x3
        )
        inception_5a_3x3 = BatchNormalization(
            axis=3, epsilon=0.00001, name="inception_5a_3x3_bn" + "2"
        )(inception_5a_3x3)
        inception_5a_3x3 = Activation("relu")(inception_5a_3x3)

        inception_5a_pool = Lambda(lambda x: x**2, name="power2_5a")(inception_4e)
        inception_5a_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception_5a_pool)
        inception_5a_pool = Lambda(lambda x: x * 9, name="mult9_5a")(inception_5a_pool)
        inception_5a_pool = Lambda(lambda x: K.sqrt(x), name="sqrt_5a")(inception_5a_pool)

        inception_5a_pool = Conv2D(96, (1, 1), strides=(1, 1), name="inception_5a_pool_conv" + "")(
            inception_5a_pool
        )
        inception_5a_pool = BatchNormalization(
            axis=3, epsilon=0.00001, name="inception_5a_pool_bn" + ""
        )(inception_5a_pool)
        inception_5a_pool = Activation("relu")(inception_5a_pool)
        inception_5a_pool = ZeroPadding2D(padding=(1, 1))(inception_5a_pool)

        inception_5a_1x1 = Conv2D(256, (1, 1), strides=(1, 1), name="inception_5a_1x1_conv" + "")(
            inception_4e
        )
        inception_5a_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_5a_1x1_bn" + "")(
            inception_5a_1x1
        )
        inception_5a_1x1 = Activation("relu")(inception_5a_1x1)

        inception_5a = concatenate([inception_5a_3x3, inception_5a_pool, inception_5a_1x1], axis=3)

        # inception_5b
        inception_5b_3x3 = Conv2D(96, (1, 1), strides=(1, 1), name="inception_5b_3x3_conv" + "1")(
            inception_5a
        )
        inception_5b_3x3 = BatchNormalization(
            axis=3, epsilon=0.00001, name="inception_5b_3x3_bn" + "1"
        )(inception_5b_3x3)
        inception_5b_3x3 = Activation("relu")(inception_5b_3x3)
        inception_5b_3x3 = ZeroPadding2D(padding=(1, 1))(inception_5b_3x3)
        inception_5b_3x3 = Conv2D(384, (3, 3), strides=(1, 1), name="inception_5b_3x3_conv" + "2")(
            inception_5b_3x3
        )
        inception_5b_3x3 = BatchNormalization(
            axis=3, epsilon=0.00001, name="inception_5b_3x3_bn" + "2"
        )(inception_5b_3x3)
        inception_5b_3x3 = Activation("relu")(inception_5b_3x3)

        inception_5b_pool = MaxPooling2D(pool_size=3, strides=2)(inception_5a)

        inception_5b_pool = Conv2D(96, (1, 1), strides=(1, 1), name="inception_5b_pool_conv" + "")(
            inception_5b_pool
        )
        inception_5b_pool = BatchNormalization(
            axis=3, epsilon=0.00001, name="inception_5b_pool_bn" + ""
        )(inception_5b_pool)
        inception_5b_pool = Activation("relu")(inception_5b_pool)

        inception_5b_pool = ZeroPadding2D(padding=(1, 1))(inception_5b_pool)

        inception_5b_1x1 = Conv2D(256, (1, 1), strides=(1, 1), name="inception_5b_1x1_conv" + "")(
            inception_5a
        )
        inception_5b_1x1 = BatchNormalization(axis=3, epsilon=0.00001, name="inception_5b_1x1_bn" + "")(
            inception_5b_1x1
        )
        inception_5b_1x1 = Activation("relu")(inception_5b_1x1)

        inception_5b = concatenate([inception_5b_3x3, inception_5b_pool, inception_5b_1x1], axis=3)

        av_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(inception_5b)
        reshape_layer = Flatten()(av_pool)
        dense_layer = Dense(128, name="dense_layer")(reshape_layer)
        norm_layer = Lambda(lambda x: K.l2_normalize(x, axis=1), name="norm_layer")(dense_layer)

        model = Model(inputs=[myInput], outputs=norm_layer)

        home = get_deepface_home()
        output = home + "/.deepface/weights/openface_weights.h5"

        if isfile() != True:
            download(url, output, quiet=False)
        
        model.load_weights(output)
        return model
