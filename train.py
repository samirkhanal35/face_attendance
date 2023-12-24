import os
import cv2
import numpy as np
from sklearn.svm import SVC
import joblib
import mediapipe as mp

# Initialize MediaPipe's FaceMesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Function to detect facial landmarks using MediaPipe


def detect_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    landmarks = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for lm in face_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])  # x, y, and z coordinates
    return landmarks

# Function to load images, detect landmarks, and collect data


def load_data_from_directory(directory):
    images = []
    landmarks_list = []
    labels = []

    for subfolder in os.listdir(directory):
        subfolder_path = os.path.join(directory, subfolder)

        for filename in os.listdir(subfolder_path):
            img_path = os.path.join(subfolder_path, filename)
            img = cv2.imread(img_path)

            if img is not None:
                # Detect landmarks
                landmarks = detect_landmarks(img)

                if landmarks:
                    landmarks_list.append(np.array(landmarks).flatten())
                    images.append(img)
                    labels.append(subfolder)

    return images, landmarks_list, labels


def train_model():
    # Load data from the raw directory
    dest_dir = os.getcwd()
    raw_dir = os.path.join(dest_dir, "raw")
    images, landmarks_list, labels = load_data_from_directory(raw_dir)

    # Convert landmarks to numpy array
    landmarks_array = np.array(landmarks_list)
    print("Landmarks_array:", landmarks_array)

    # Initialize and train the SVC classifier
    clf = SVC(kernel='linear', C=1.0, probability=True)
    clf.fit(landmarks_array, labels)

    # Save the trained model to the 'model' directory
    model_dir = os.path.join(dest_dir, "model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, "svm_classifier.pkl")
    joblib.dump(clf, model_path)

    print(f"Trained model saved to {model_path}")
