import cv2
import sqlite3
import os
import time
import numpy as np
from datetime import datetime

# Utility function to try different camera ports
def try_camera_ports():
    """Try different camera ports and return the first working one."""
    for port in [0, 1, -1]:  # Try ports 0, 1, and default
        print(f"Trying camera port {port}...")
        cap = cv2.VideoCapture(port, cv2.CAP_DSHOW)  # Add cv2.CAP_DSHOW for Windows
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Successfully connected to camera on port {port}")
                return cap
            cap.release()
    return None

# Check if model and cascade files exist
cascade_path = "haarcascade/haarcascade_frontalface_default.xml"
model_path = "recognizer/trainingdata.yml"

if not os.path.exists(cascade_path):
    raise FileNotFoundError(f"Cascade classifier file not found at {cascade_path}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Initialize face detection and recognition
facedetect = cv2.CascadeClassifier(cascade_path)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)

# Configuration
CONFIDENCE_THRESHOLD = 65
SCALE_FACTOR = 1.2
MIN_NEIGHBORS = 5

def get_profile(id):
    """Fetch user profile based on the ID from the database."""
    try:
        conn = sqlite3.connect("FaceBase.db")
        cursor = conn.execute("SELECT * FROM Peoples WHERE id=?", (id,))
        profile = cursor.fetchone()
        return profile
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None
    finally:
        conn.close()

def detect_faces(image_path):
    """Detect faces in an image and return a list of workers."""
    face_cascade = cv2.CascadeClassifier(cascade_path)
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, SCALE_FACTOR, MIN_NEIGHBORS)
    
    workers = []
    for (x, y, w, h) in faces:
        workers.append('Worker')  # Placeholder for worker name
    return workers

def detect_face_attributes(image):
    """Detect faces and their attributes in the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    face_data = []
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        
        # Recognize face
        id, confidence = recognizer.predict(face)
        
        if confidence < 100:
            conn = sqlite3.connect("FaceBase.db")
            cursor = conn.cursor()
            cursor.execute("SELECT Name, Age, Gender FROM Peoples WHERE id = ?", (id,))
            profile = cursor.fetchone()
            
            if profile:
                name, age, gender = profile
                cursor.execute("INSERT INTO Detections (person_id) VALUES (?)", (id,))
                conn.commit()
                
                face_data.append({
                    'id': id,
                    'name': name,
                    'age': age,
                    'gender': gender,
                    'confidence': confidence,
                    'box': (x, y, w, h)
                })
            conn.close()
    
    return face_data

def insert_detection(user_id):
    """Insère une détection dans la base de données."""
    try:
        conn = sqlite3.connect("FaceBase.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO detections (user_id, detection_time) VALUES (?, ?)", (user_id, datetime.now()))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Erreur lors de l'insertion de la détection: {e}")
    finally:
        conn.close()

def generate_frames():
    """Generate video frames with face recognition annotations."""
    camera = try_camera_ports()
    if not camera:
        print("No camera found!")
        return
        
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            face_data = detect_face_attributes(frame)
            
            # Draw rectangles and labels on detected faces
            for face in face_data:
                x, y, w, h = face['box']
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Create label with name, age, and gender
                label = f"{face['name']} ({face['age']}, {face['gender']})"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def main():
    """Main function to initialize the camera and start the face recognition loop."""
    try:
        print("Initializing camera...")
        cam = try_camera_ports()
        
        if cam is None:
            raise Exception("No working camera found. Please check your camera connection.")

        print("Camera initialized successfully! Starting face detection... Press 'q' to quit")

        while True:
            ret, img = cam.read()
            if not ret:
                print("Failed to read frame, retrying...")
                time.sleep(0.1)
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, SCALE_FACTOR, MIN_NEIGHBORS)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face_roi = gray[y:y+h, x:x+w]
                
                try:
                    id, conf = recognizer.predict(face_roi)
                    
                    if conf < CONFIDENCE_THRESHOLD:
                        profile = get_profile(id)
                        if profile:
                            info_color = (0, 255, 127)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(img, f"Name: {profile[1]}", (x, y+h+30), font, 0.8, info_color, 2)
                            cv2.putText(img, f"Age: {profile[2]}", (x, y+h+60), font, 0.8, info_color, 2)
                            cv2.putText(img, f"Gender: {profile[3]}", (x, y+h+90), font, 0.8, info_color, 2)
                            insert_detection(id)  # Insert detection into database
                    else:
                        cv2.putText(img, "Unknown", (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                except Exception as e:
                    print(f"Recognition error: {e}")

            cv2.imshow("Face Recognition", img)
            if cv2.waitKey(1) == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'cam' in locals() and cam is not None:
            cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
