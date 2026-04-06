import os
import cv2
import torch
import pickle
import numpy as np
import mediapipe as mp

from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

mp_face_detection = mp.solutions.face_detection

# Load FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

dataset_path = "Images"

embeddings = []
labels = []

print("Generating embeddings...")

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

    for person in os.listdir(dataset_path):

        person_folder = os.path.join(dataset_path, person)

        if not os.path.isdir(person_folder):
            continue

        for image_name in os.listdir(person_folder):

            img_path = os.path.join(person_folder, image_name)

            img = cv2.imread(img_path)

            if img is None:
                continue

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = face_detection.process(imgRGB)

            if results.detections:

                h, w, _ = img.shape
                bbox = results.detections[0].location_data.relative_bounding_box

                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                face = imgRGB[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                face = transform(face).unsqueeze(0)

                with torch.no_grad():
                    embedding = model(face).numpy()[0]

                embeddings.append(embedding)
                labels.append(person)

                print("Processed:", img_path)

# Save embeddings
data = {"embeddings": embeddings, "labels": labels}

with open("embeddings.pickle", "wb") as f:
    pickle.dump(data, f)

print("Embeddings saved.")