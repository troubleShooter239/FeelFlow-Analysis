from sys import argv
from json import dumps

from deepface.DeepFace import analyze

ACCESSED_METHODS = [
    "face_analyze"
]

args = argv[1:]


def face_analyze(img_path: str) -> str:
    try:
        res = dumps(analyze(img_path=img_path), indent=2)
        return res
    except FileNotFoundError as e:
        return dumps({"Error": f"File not found: {img_path}"})
    except Exception as e:
        return dumps({"Error": f"Error analyzing image: {e}"})


if __name__ == "__main__":
    func = args[0]
    func_args = args[1:]

    if func not in ACCESSED_METHODS:
        print(f"Method was not foud: {func}")
        exit(1)

    print(eval(func)(*func_args))