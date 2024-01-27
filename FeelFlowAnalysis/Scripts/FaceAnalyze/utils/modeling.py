from tensorflow.python.keras.models import Model

from extended_models import face_attributes
from models import recognition_models


def build_model(model_name: str) -> Model:
    """This function builds a deepface model
    Parameters:
        model_name (string): face recognition or facial attribute model
        VGG-Face, Facenet, OpenFace, DeepFace, DeepID for face recognition
        Age, Gender, Emotion, Race for facial attributes

    Returns:
            built deepface model ( (tf.)keras.models.Model )"""
    global model_obj
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
    if "model_obj" not in globals():
        model_obj = dict()
    if model_name not in model_obj[model_name]:
        model_obj[model_name] = models.get(model_name)()
    return model_obj[model_name]