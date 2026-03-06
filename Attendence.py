import cv2
import numpy as np
import mediapipe as mp
import os
from datetime import datetime as d

# MediaPipe face detection setup
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

path='Images'
imgs=[]
classNames=[]
known_faces = {}

l=os.listdir(path)
print(l)

# Load images and train face recognition
recognizer = cv2.face.LBPHFaceRecognizer_create()
trained_images = []
trained_labels = []
label_map = {}

for idx, i in enumerate(l):
    img_path = f'{path}/{i}'
    if os.path.isfile(img_path):
        curimg = cv2.imread(img_path)
        if curimg is not None:
            gray = cv2.cvtColor(curimg, cv2.COLOR_BGR2GRAY)
            trained_images.append(gray)
            class_name = os.path.splitext(i)[0]
            trained_labels.append(idx)
            label_map[idx] = class_name
            classNames.append(class_name)
            print(f'Loaded: {class_name}')

print(classNames)

# Train the recognizer
if trained_images:
    recognizer.train(trained_images, np.array(trained_labels))
    print('Training complete!')
else:
    print('No images found in Images folder')

def attendence(name):
    f=open('Attendence.csv','r+')
    namelisto=[]
    r=f.readlines()
    for i in r:
        entry=i.split(',')
        namelisto.append(entry[0])
    if name not in namelisto:
        t=d.now()
        time=t.strftime('%H:%M:%S')
        date=t.strftime('%d/%b/%Y')
        f.writelines(f'\n{name},{time},{date}')

print('Done!')   #SHOWS TRAINING ARE FINISHED

cam=cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while True:
        success, img = cam.read()
        if not success:
            break
        
        h, w, c = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(imgRGB)
        
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x1 = int(bboxC.xmin * w)
                y1 = int(bboxC.ymin * h)
                x2 = int((bboxC.xmin + bboxC.width) * w)
                y2 = int((bboxC.ymin + bboxC.height) * h)
                
                # Ensure coordinates are within bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Extract face region
                face_roi = img[y1:y2, x1:x2]
                if face_roi.size > 0:
                    gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    gray_roi = cv2.resize(gray_roi, (200, 200))
                    
                    # Recognize face using LBPH
                    label, confidence = recognizer.predict(gray_roi)
                    
                    # If confidence is low (below 70), it's a recognized face
                    if confidence < 70 and label in label_map:
                        name = label_map[label].upper()
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, f'{name} ({int(confidence)})', (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                        attendence(name)
                    else:
                        # Unknown face
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(img, 'Unknown', (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()





