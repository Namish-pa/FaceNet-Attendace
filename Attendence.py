import cv2
import numpy as np
import mediapipe as mp
import os
from datetime import datetime

mp_face_detection = mp.solutions.face_detection

path = "Images"

trained_images = []
trained_labels = []
label_map = {}
classNames = []
label_id = 0

print("Loading training images...")

# ==========================
# LOAD DATASET
# ==========================

for person_name in os.listdir(path):

    person_folder = os.path.join(path, person_name)

    if not os.path.isdir(person_folder):
        continue

    label_map[label_id] = person_name
    classNames.append(person_name)

    for image_name in os.listdir(person_folder):

        img_path = os.path.join(person_folder, image_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # normalize lighting
        gray = cv2.equalizeHist(gray)

        gray = cv2.resize(gray,(200,200))

        # flipped version
        flipped = cv2.flip(gray,1)

        trained_images.append(gray)
        trained_images.append(flipped)

        trained_labels.append(label_id)
        trained_labels.append(label_id)

        print("Loaded:", img_path)

    label_id += 1

print("Total training images:", len(trained_images))

# ==========================
# TRAIN MODEL
# ==========================

recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=2,
    neighbors=16,
    grid_x=8,
    grid_y=8
)

recognizer.train(trained_images, np.array(trained_labels))

print("Training complete!")

# ==========================
# ATTENDANCE FUNCTION
# ==========================

def attendence(name):

    if not os.path.exists("Attendence.csv"):
        with open("Attendence.csv","w") as f:
            f.write("Name,Time,Date\n")

    with open("Attendence.csv","r+") as f:

        lines = f.readlines()
        name_list = []

        for line in lines:
            entry = line.split(",")
            name_list.append(entry[0])

        if name not in name_list:

            now = datetime.now()

            time = now.strftime("%H:%M:%S")
            date = now.strftime("%d/%b/%Y")

            f.writelines(f"\n{name},{time},{date}")

print("Starting webcam...")

# ==========================
# CAMERA
# ==========================

cam = cv2.VideoCapture(0)

cam.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

# ==========================
# RECOGNITION LOOP
# ==========================

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

    while True:

        ret, img = cam.read()

        if not ret:
            print("Frame failed")
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

                face = img[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                gray = cv2.equalizeHist(gray)

                gray = cv2.resize(gray,(200,200))

                label, confidence = recognizer.predict(gray)

                if confidence < 60 and label in label_map:

                    name = label_map[label].upper()

                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

                    cv2.putText(img,f"{name} ({int(confidence)})",
                                (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9,
                                (255,255,255),
                                2)

                    attendence(name)

                else:

                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)

                    cv2.putText(img,"Unknown",
                                (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9,
                                (0,0,255),
                                2)

        cv2.imshow("Face Attendance",img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cam.release()
cv2.destroyAllWindows()