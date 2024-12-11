import cv2
import sqlite3
import os
import time

def try_camera_ports():
    """Try different camera ports and return the first working one"""
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

def create_dataset_dir():
    if not os.path.exists("dataset"):
        os.makedirs("dataset")
        print("Dataset directory created")

def validate_input(Id, Name, Age, Gen):
    if not Id.isdigit():
        raise ValueError("ID must be a number")
    if not Name.strip():
        raise ValueError("Name cannot be empty")
    try:
        age = int(Age)
        if age < 0 or age > 120:
            raise ValueError("Age must be between 0 and 120")
    except ValueError:
        raise ValueError("Age must be a valid number")
    if not Gen.strip():
        raise ValueError("Gender cannot be empty")

def insertOrUpdate(Id, Name, Age, Gen):
    try:
        conn = sqlite3.connect("FaceBase.db")
        conn.execute('''CREATE TABLE IF NOT EXISTS Peoples
            (id INTEGER PRIMARY KEY,
            Name TEXT NOT NULL,
            Age INTEGER,
            Gender TEXT)''')
        
        cursor = conn.execute("SELECT * FROM Peoples WHERE id=?", (Id,))
        isRecordExist = cursor.fetchone() is not None

        if isRecordExist:
            conn.execute("UPDATE Peoples SET Name=?, Age=?, Gender=? WHERE id=?", (Name, Age, Gen, Id))
            print("User information updated in database")
        else:
            conn.execute("INSERT INTO Peoples (id, Name, Age, Gender) VALUES (?, ?, ?, ?)", (Id, Name, Age, Gen))
            print("New user added to database")

        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        raise
    finally:
        conn.close()

def insert_user(Id, Age, Gen, name):
    try:
        conn = sqlite3.connect("FaceBase.db")
        conn.execute('''CREATE TABLE IF NOT EXISTS users
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL)''')
        conn.execute("INSERT INTO users (name) VALUES (?)", (name,))
        conn.commit()
        print(f"User '{name}' added to users table.")
        insertOrUpdate(Id, name, Age, Gen)
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()

def main():
    try:
        # Create dataset directory if it doesn't exist
        create_dataset_dir()

        # Load face detection cascade
        cascade_path = "haarcascade/haarcascade_frontalface_default.xml"
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Cascade classifier file not found at {cascade_path}")
        
        faceDetect = cv2.CascadeClassifier(cascade_path)

        # Get user input
        Id = input('Enter User ID: ')
        name = input('Enter User Name: ')
        age = input('Enter User Age: ')
        gen = input('Enter User Gender: ')

        # Validate input
        validate_input(Id, name, age, gen)

        # Update database
        insert_user(Id, age, gen, name)

        # Initialize camera with improved detection
        print("\nInitializing camera...")
        cam = try_camera_ports()
        
        if cam is None:
            raise Exception("No working camera found. Please check your camera connection.")

        print("Camera initialized successfully!")

        # Configuration
        SAMPLE_COUNT = 20
        sampleNum = 0
        last_capture_time = time.time()
        CAPTURE_DELAY = 0.5  # Seconds between captures

        print(f"\nStarting face capture. Please look at the camera...")
        print(f"Press 'q' to quit or wait until {SAMPLE_COUNT} samples are captured")

        while True:
            ret, img = cam.read()
            if not ret:
                print("Failed to read frame, retrying...")
                time.sleep(0.1)  # Add small delay before retry
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceDetect.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                current_time = time.time()
                if current_time - last_capture_time >= CAPTURE_DELAY:
                    sampleNum += 1
                    # Save face image
                    face_filename = f"dataset/User.{Id}.{sampleNum}.jpg"
                    cv2.imwrite(face_filename, gray[y:y+h, x:x+w])
                    print(f"Captured sample {sampleNum}/{SAMPLE_COUNT}")
                    last_capture_time = current_time

                    # Insert face image path into UserFaces table
                    conn = sqlite3.connect("FaceBase.db")
                    conn.execute('''CREATE TABLE IF NOT EXISTS UserFaces (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        face_image_path TEXT NOT NULL,
                        FOREIGN KEY (user_id) REFERENCES Peoples(id)
                    );''')
                    conn.execute("INSERT INTO UserFaces (user_id, face_image_path) VALUES (?, ?)", (Id, face_filename))
                    conn.commit()
                    conn.close()
                    print(f"Face image '{face_filename}' added to UserFaces table.")

                # Draw rectangle around face
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Show progress
                cv2.putText(img, f"Progress: {sampleNum}/{SAMPLE_COUNT}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Face Capture", img)
            if cv2.waitKey(1) == ord('q') or sampleNum >= SAMPLE_COUNT:
                break

        print("\nFace capture completed successfully!")
        print(f"Captured {sampleNum} samples for user {name}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'cam' in locals() and cam is not None:
            cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
