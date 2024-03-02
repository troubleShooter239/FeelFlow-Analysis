import models.face_attributes as face_attributes
import models.recognition_models as recognition_models
from utils.functions import get_tf_major_version

if get_tf_major_version() == 1:
    from keras.models import Model
else:
    from tensorflow.keras.models import Model


def build_model(model_name: str) -> Model:
    global model_obj
    if not "model_obj" in globals():
        model_obj = dict()
    models = {
        "VGG-Face": recognition_models.VggFaceClient,
        "OpenFace": recognition_models.OpenFaceClient,
        "Facenet": recognition_models.FaceNet128dClient,
        "Facenet512": recognition_models.FaceNet512dClient,
        "DeepFace": recognition_models.DeepFaceClient,
        "DeepID": recognition_models.DeepIdClient,
        "Dlib": recognition_models.DlibClient,
        "ArcFace": recognition_models.ArcFaceClient,
        "SFace": recognition_models.SFaceClient,
        "Emotion": face_attributes.EmotionClient,
        "Age": face_attributes.ApparentAgeClient,
        "Gender": face_attributes.GenderClient,
        "Race": face_attributes.RaceClient
    }
    if not model_name in model_obj:
        model_obj[model_name] = models[model_name]()
    return model_obj[model_name]
