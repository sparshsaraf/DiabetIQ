# modules/face_module.py
from deepface import DeepFace
import os

FACES_DIR = os.path.join(os.getcwd(), "uploads", "faces_db")

def find_match(img_path):
    try:
        result = DeepFace.find(
            img_path=img_path,
            db_path=FACES_DIR,
            enforce_detection=False
        )
        if len(result) > 0 and len(result[0]) > 0:
            matched = result[0].iloc[0]["identity"]
            return matched  # path to matched image
    except Exception as e:
        print("DeepFace error:", e)

    return None
