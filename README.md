# FaceNet Attendance System

This project is a computer vision-based attendance system using a pre-trained FaceNet model (`InceptionResnetV1`) to recognize faces and log attendance automatically.

## Features
- **Face Detection:** Uses MediaPipe for fast, real-time face detection from a webcam feed.
- **Face Recognition:** Uses FaceNet (`facenet-pytorch`) to generate embeddings for detected faces and records cosine similarity.
- **Attendance Logging:** Recognized faces are automatically logged in `Attendance.csv` with the date and time.

## Project Structure
- `train_embeddings.py` - Script to generate face embeddings for known individuals from the `Images/` directory and save them in a pickle file.
- `facenet_attendance.py` - Main script that accesses the webcam to perform real-time face recognition and logs attendance.
- `Attendance.csv` - The attendance log containing Name, Time, and Date.
- `embeddings.pickle` - The stored face embeddings.

## How to use

1. **Create and activate a virtual environment**:
   ```bash
   # Create venv
   python -m venv venv

   # Activate venv (Windows)
   .\venv\Scripts\activate

   # Activate venv (macOS/Linux)
   source venv/bin/activate
   ```
2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. Place pictures of the people you want to recognize in the `Images/` folder. Use subfolders for each person's name (e.g., `Images/Person_Name/pic1.jpg`).
4. Run `python train_embeddings.py` to process those images and save their embeddings locally.
5. Run `python facenet_attendance.py` to launch the live facial recognition tracker using your webcam.
6. Press `q` to quit the webcam application.
