from collections import defaultdict

from deepface.basemodels import VGGFace, OpenFace, FbDeepFace, DeepID, ArcFace, SFace, Dlib, Facenet
from deepface.extendedmodels import Age, Gender, Race, Emotion
from keras.models import Model


def build_model(model_name: str) -> Model:
    """
    This function builds a deepface model
    Parameters:
        model_name (string): face recognition or facial attribute model
        VGG-Face, Facenet, OpenFace, DeepFace, DeepID for face recognition
        Age, Gender, Emotion, Race for facial attributes

    Returns:
            built deepface model ( (tf.)keras.models.Model )
    """
    global model_obj
    model_obj = defaultdict(dict)

    models = {
        "VGG-Face": VGGFace.VggFaceClient,
        "OpenFace": OpenFace.OpenFaceClient,
        "Facenet": Facenet.FaceNet128dClient,
        "Facenet512": Facenet.FaceNet512dClient,
        "DeepFace": FbDeepFace.DeepFaceClient,
        "DeepID": DeepID.DeepIdClient,
        "Dlib": Dlib.DlibClient,
        "ArcFace": ArcFace.ArcFaceClient,
        "SFace": SFace.SFaceClient,
        "Emotion": Emotion.EmotionClient,
        "Age": Age.ApparentAgeClient,
        "Gender": Gender.GenderClient,
        "Race": Race.RaceClient,
    }

    if model_name not in model_obj[model_name]:
        model_obj[model_name] = models.get(model_name)()

    return model_obj[model_name]