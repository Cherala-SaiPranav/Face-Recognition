import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# Initialize known faces database
known_face_encodings = []
known_face_names = []

# Load known face images and create encodings (use file names as labels)
def load_known_faces(known_faces_dir=r"C:\Users\saipr\OneDrive\Desktop\New folder\face-recog"):
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(known_faces_dir, filename)
            image = cv2.imread(image_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                # Extract the landmarks as an encoding (normalized x, y coordinates)
                landmarks = [(lm.x, lm.y) for lm in face_landmarks.landmark]
                known_face_encodings.append(landmarks)
                known_face_names.append(os.path.splitext(filename)[0])  # File name as label

# Load faces
load_known_faces()

# Helper function to recognize face based on known encodings
def recognize_face(face_encoding):
    distances = [np.linalg.norm(face_encoding - known_face) for known_face in known_face_encodings]
    min_distance_index = np.argmin(distances)
    if distances[min_distance_index] < 0.5:  # Adjust threshold based on testing
        return known_face_names[min_distance_index]
    return "Unknown"

# Start webcam and process video frames
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the frame
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            # Extract face landmarks as a feature
            face_encoding = np.array([(lm.x, lm.y) for lm in landmarks.landmark])
            
            # Recognize the face
            name = recognize_face(face_encoding)
            
            # Draw a box around the face (bounding box can be added for visualization)
            h, w, _ = frame.shape
            # You can calculate bounding box based on landmarks if needed
            cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Face Recognition", frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()
