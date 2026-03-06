import cv2

cam = cv2.VideoCapture(0)

print("Camera opened:", cam.isOpened())

while True:
    ret, frame = cam.read()

    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()