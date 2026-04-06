import cv2
import pickle
import torch
import numpy as np
import mediapipe as mp
from datetime import datetime

from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity

# ==========================
# Load FaceNet Model
# ==========================

model = InceptionResnetV1(pretrained='vggface2').eval()

# ==========================
# Image Preprocessing
# ==========================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160,160)),
    transforms.ToTensor()
])

# ==========================
# Load Stored Embeddings
# ==========================

print("Loading embeddings...")

data = pickle.load(open("embeddings.pickle","rb"))

known_embeddings = np.array(data["embeddings"])
known_labels = data["labels"]

print("Embeddings loaded:", len(known_embeddings))

# ==========================
# MediaPipe Face Detector
# ==========================

mp_face_detection = mp.solutions.face_detection

# ==========================
# Attendance Function
# ==========================

def mark_attendance(name):

    try:
        with open("Attendance.csv","r+") as f:

            lines = f.readlines()
            names = [line.split(",")[0] for line in lines]

            if name not in names:

                now = datetime.now()
                time = now.strftime("%H:%M:%S")
                date = now.strftime("%d/%b/%Y")

                f.write(f"\n{name},{time},{date}")

    except:
        with open("Attendance.csv","w") as f:
            f.write("Name,Time,Date\n")

# ==========================
# Start Webcam
# ==========================

cam = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

    while True:

        ret, img = cam.read()

        if not ret:
            continue

        h, w, _ = img.shape

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = face_detection.process(imgRGB)

        if results.detections:

            for detection in results.detections:

                bbox = detection.location_data.relative_bounding_box

                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)

                x1, y1 = max(0,x1), max(0,y1)
                x2, y2 = min(w,x2), min(h,y2)

                face = imgRGB[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                face_input = transform(face).unsqueeze(0)

                with torch.no_grad():
                    embedding = model(face_input).numpy()

                # Compare embeddings
                similarities = cosine_similarity(embedding, known_embeddings)

                best_index = np.argmax(similarities)

                score = similarities[0][best_index]

                # Recognition threshold
                if score > 0.6:

                    name = known_labels[best_index].upper()

                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

                    cv2.putText(img,
                                f"{name} ({score:.2f})",
                                (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9,
                                (255,255,255),
                                2)

                    mark_attendance(name)

                else:

                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)

                    cv2.putText(img,
                                "Unknown",
                                (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9,
                                (0,0,255),
                                2)

        cv2.imshow("FaceNet Attendance System", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()