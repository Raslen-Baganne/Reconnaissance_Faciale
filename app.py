from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import sqlite3
from datetime import datetime
import os
import time
import logging

app = Flask(__name__)

def get_db_connection():
    conn = sqlite3.connect('FaceBase.db')
    conn.row_factory = sqlite3.Row
    return conn

# Initialisation du détecteur de visages
face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')

# Vérifier si 'cv2.face' est disponible pour le reconnaisseur
if not hasattr(cv2, 'face'):
    raise ImportError("Le module 'cv2.face' n'est pas disponible. Veuillez installer opencv-contrib-python.")

recognizer = cv2.face.LBPHFaceRecognizer_create()

# Charger le modèle entraîné s'il existe
try:
    recognizer.read('recognizer/trainingdata.yml')
except:
    print("Aucun modèle d'entraînement trouvé")

def get_camera():
    # Try different camera backends
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    
    for backend in backends:
        for index in range(2):  # Try first two camera indices
            try:
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    # Set camera properties for better performance
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    return cap
            except Exception as e:
                print(f"Failed to open camera {index} with backend {backend}: {str(e)}")
                if cap:
                    cap.release()
    return None

def generate_frames():
    camera = get_camera()
    if camera is None:
        print("Erreur: Impossible d'ouvrir la caméra. Vérifiez que la caméra est connectée et n'est pas utilisée par une autre application.")
        return

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Erreur: Impossible de lire l'image de la caméra")
                camera.release()
                time.sleep(1)
                camera = get_camera()
                if camera is None:
                    break
                continue
            
            # Détection et reconnaissance des visages
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    # Reconnaissance du visage
                    roi_gray = gray[y:y+h, x:x+w]
                    try:
                        id_, conf = recognizer.predict(roi_gray)
                        if conf >= 45 and conf <= 85:
                            conn = get_db_connection()
                            cursor = conn.cursor()
                            
                            # Récupérer le nom de la personne depuis la base de données
                            cursor.execute("SELECT name FROM users WHERE id = ?", (id_,))
                            result = cursor.fetchone()
                            if result:
                                name = result['name']
                                # Ajouter la détection à l'historique
                                now = datetime.now()
                                cursor.execute(""" 
                                    INSERT INTO detections (user_id, detection_time)
                                    VALUES (?, ?)
                                """, (id_, now))
                                conn.commit()
                                
                                # Afficher le nom sur le cadre
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                cv2.putText(frame, name, (x+2,y+h-5), font, 1, (150,255,0), 2)
                            conn.close()
                    except Exception as e:
                        print(f"Erreur lors de la prédiction: {str(e)}")
                        continue

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"Erreur lors du traitement de l'image: {str(e)}")
                continue
    except Exception as e:
        print(f"Erreur générale: {str(e)}")
    finally:
        if camera:
            camera.release()

@app.route('/')
def index():
    conn = get_db_connection()
    cursor = conn.cursor()
    # Récupérer l'historique des détections avec les noms des utilisateurs
    cursor.execute("""
        SELECT users.name, detections.detection_time 
        FROM detections 
        JOIN users ON detections.user_id = users.id 
        ORDER BY detections.detection_time DESC 
        LIMIT 10
    """)
    detections = cursor.fetchall()
    logging.info(f'Détections récupérées: {detections}')
    conn.close()
    return render_template('index.html', detections=detections)

@app.route('/get_latest_detections')
def get_latest_detections():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT users.name, detections.detection_time
            FROM detections 
            JOIN users ON detections.user_id = users.id 
            ORDER BY detections.detection_time DESC 
            LIMIT 10
        """)
        detections = cursor.fetchall()
        logging.info(f'Détections récupérées: {detections}')
        return jsonify([{
            'name': detection['name'],
            'detection_time': detection['detection_time']
        } for detection in detections])

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Créer les tables si elles n'existent pas
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(""" 
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute(""" 
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            detection_time TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    
    conn.commit()
    conn.close()
    
    app.run(debug=True)
