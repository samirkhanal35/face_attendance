import os
import cv2
import numpy as np
from sklearn.svm import SVC
import joblib

# Function to load images and their labels


def load_data_from_directory(directory):
    images = []
    labels = []

    for subfolder in os.listdir(directory):
        subfolder_path = os.path.join(directory, subfolder)

        for filename in os.listdir(subfolder_path):
            img_path = os.path.join(subfolder_path, filename)
            img = cv2.imread(img_path)

            if img is not None:
                # Preprocess the image (e.g., resize)
                img = cv2.resize(img, (128, 128))  # Resize to 128x128
                images.append(img.flatten())
                labels.append(subfolder)

    return np.array(images), np.array(labels)


def train_model():
    # Load data from the raw directory
    dest_dir = os.getcwd()
    raw_dir = os.path.join(dest_dir, "raw")
    images, labels = load_data_from_directory(raw_dir)

    # Initialize and train the SVC classifier
    clf = SVC(kernel='linear', C=1.0, probability=True)
    clf.fit(images, labels)

    # Save the trained model to the 'model' directory
    model_dir = os.path.join(dest_dir, "model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, "svm_classifier.pkl")
    joblib.dump(clf, model_path)

    print(f"Trained model saved to {model_path}")
