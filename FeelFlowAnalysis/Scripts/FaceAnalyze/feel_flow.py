from os import path, stat
from time import ctime, time
from typing import Any, Dict, List, Tuple, Union
from json import dumps, loads

import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
from PIL.TiffImagePlugin import IFDRational
from tensorflow.keras.layers import Model

import commons.functions as F
from commons.distance import find_cosine, find_euclidean
from commons.folder_utils import initialize_folder
from commons.face_processor import processor_methods

initialize_folder()


def analyze(img: Union[str, np.ndarray], 
            actions: str = '{"age": true, "emotion": true, "gender": true, "race": true}',
            enforce_detection: bool = True, align: bool = True) -> str:
    try:
        img_objs = F.extract_faces(img, (224, 224), False, enforce_detection, align)
    except ValueError:
        return "{}"
    models: Dict[str, Model] = {a: F.build_model(a.capitalize()) for a, s in loads(actions).items() if s}
    resp_objects = []
    # TODO: Make it parallel
    for img, region, confidence in img_objs:
        if img.shape[0] <= 0 or img.shape[1] <= 0: 
            continue
        obj = {"region": region, "face_confidence": confidence}
        for action, model in models.items():
            try:
                obj.update(processor_methods[action](model.predict(img)))
            except Exception:
                continue

        resp_objects.append(obj)

    return dumps(resp_objects, indent=2)


def verify(img1: Union[str, np.ndarray], img2: Union[str, np.ndarray], 
           model_name: str = "VGG-Face", distance_metric: str = "cosine", 
           enforce_detection: bool = True, align: bool = True, 
           normalization: str = "base") -> Dict[str, Any]:
    target_size = F.find_size(model_name)

    distances, regions = [], []
    for c1, r1, _ in F.extract_faces(img1, target_size, False, enforce_detection, align):
        for c2, r2, _ in F.extract_faces(img2, target_size, False, enforce_detection, align):
            repr1 = F.represent(c1, model_name, enforce_detection, "skip", 
                              align, normalization)[0]["embedding"]
            repr2 = F.represent(c2, model_name, enforce_detection, "skip", 
                              align, normalization)[0]["embedding"]

            if distance_metric == "cosine":
                dst = find_cosine(repr1, repr2)
            elif distance_metric == "euclidean":
                dst = find_euclidean(repr1, repr2)
            else:
                dst = find_euclidean(dst.l2_normalize(repr1), dst.l2_normalize(repr2))

            distances.append(dst)
            regions.append((r1, r2))

    threshold = F.find_threshold(model_name, distance_metric)
    distance = min(distances)
    facial_areas = regions[np.argmin(distances)]
    return {
        "verified": True if distance <= threshold else False,
        "distance": distance,
        "threshold": threshold,
        "model": model_name,
        "similarity_metric": distance_metric,
        "facial_areas": {"img1": facial_areas[0], "img2": facial_areas[1]}
    }


# def get_gps(exif_data) -> Tuple:
#     if 'GPSInfo' in exif_data:
#         gps_info = exif_data['GPSInfo']
#         lat_ref = gps_info[1]
#         lat = gps_info[2]
#         lon_ref = gps_info[3]
#         lon = gps_info[4]
#         lat_deg = lat[0][0] / float(lat[0][1])
#         lat_min = lat[1][0] / float(lat[1][1])
#         lat_sec = lat[2][0] / float(lat[2][1])
#         lon_deg = lon[0][0] / float(lon[0][1])
#         lon_min = lon[1][0] / float(lon[1][1])
#         lon_sec = lon[2][0] / float(lon[2][1])
#         if lat_ref == 'S':
#             lat_deg = -lat_deg
#         if lon_ref == 'W':
#             lon_deg = -lon_deg
#         return lat_deg, lat_min, lat_sec, lon_deg, lon_min, lon_sec
#     return ()
# from PIL import Image
# import piexif


# img = Image.open("D:/1.jpg")
# exif_dict = piexif.load(img.info['exif'])
# piexif.GPSIFD.GPSLatitude
# if 'GPS' in exif_dict:
#     gps = exif_dict['GPS']
#     print(gps)
#     latitude = gps[2][0][0] / float(gps[2][0][1])
#     longitude = gps[4][0][0] / float(gps[4][0][1])
#     print('Широта:', latitude)
#     print('Долгота:', longitude)
# else:
#     print('Геоданные отсутствуют')


# TODO: Converting to ndarray is bring's losing img information
def get_image_metadata(image: str) -> str:
    with Image.open(image) as i:
        try:
            mime = Image.MIME[i.format]
        except KeyError:
            mime = None
        data = {
            "image_size": i.size,
            "file_type": i.format,
            "mime": mime,
            "info?": i.info,
            #"time_created": ctime(path.getctime(i)),
            #"name": path.basename(i),
            #"size": path.getsize(i),
            #"access_time": ctime(path.getatime(i)),
            #"location": path.abspath(i),
            #"inode_change_time": ctime(path.getctime(i)),
            #"permission": oct(stat(i).st_mode)[-4:],
            #"type_extension": path.splitext(i)[1][1:],
            "band_names": i.getbands(),
            "bbox": i.getbbox(),
            #"icc_profile": i.info["icc_profile"],
            "megapixels": round(i.size[0] * i.size[1] / 1000000, 2),
        }
        #data["dpi"] = tuple(map(float, i.info["dpi"]))
        #exif_data = {TAGS.get(t, t): float(v) if isinstance(v, IFDRational) else v for t, v in i.getexif().items()}
        #data.update(exif_data)
        #data["GPSInfo"] = get_gps(exif_data)
    
    return dumps(data, indent=2)


import base64

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        image_binary = image_file.read()
        base64_encoded = base64.b64encode(image_binary)
        base64_string = base64_encoded.decode('utf-8')
    return base64_string

#print(encode_image_to_base64("C:/Users/lilia/OneDrive/Изображения/2.jpg"))
# print(get_image_metadata(encode_image_to_base64("C:/Users/lilia/OneDrive/Изображения/2.jpg")))

start = time()
a = analyze("C:/Users/lilia/OneDrive/Изображения/1.jpg")
print(time() - start)
print(a)
