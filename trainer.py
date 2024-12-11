import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = "dataset"

# Fonction pour récupérer les images et les IDs
def get_images_with_ids(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]
    faces = []
    ids = []
    for image_path in image_paths:
        faceImg = Image.open(image_path).convert("L")  # Convertir en niveaux de gris
        faceNp = np.array(faceImg, np.uint8)
        Id = int(os.path.split(image_path)[-1].split(".")[1])
        faces.append(faceNp)
        ids.append(Id)

    return np.array(ids), faces

# Entraîner le modèle
ids, faces = get_images_with_ids(path)
recognizer.train(faces, ids)
recognizer.save("recognizer/trainingdata.yml")
print("Training completed!")
cv2.destroyAllWindows()
