import streamlit as st
import cv2
import mediapipe as mp
import pandas as pd
from sklearn.svm import SVC
import numpy as np
import joblib

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Load the saved SVM classifier
svm_classifier = joblib.load('model/svm_classifier.pkl')


# Initialize DataFrame to store attendance
attendance_df = pd.DataFrame(columns=["ID", "Name", "Timestamp"])


# Streamlit app
def main():
    st.title("Facial Recognition Attendance System using SVC and MediaPipe")

    # Display the video feed in a specified region
    video_placeholder = st.empty()

    # Start video capture
    video_capture = cv2.VideoCapture(0)

    # stopping flag
    stop_flag = True

    # Recognition and attendance
    while stop_flag:
        _, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Use MediaPipe to find facial landmarks
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                # Convert landmarks to a numpy array (for simplicity, consider only one face in the frame)
                landmark_array = np.array(
                    [[landmark.x, landmark.y, landmark.z] for landmark in landmarks.landmark]).flatten()
                predicted_label = svm_classifier.predict([landmark_array])[0]

                # Display recognized ID on the video frame
                cv2.putText(frame, f"ID: {predicted_label}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Check if attendance already marked
                if predicted_label not in attendance_df["ID"].values:
                    attendance_df.loc[len(attendance_df)] = [
                        predicted_label, f"Student {predicted_label}", pd.Timestamp.now()]
                    st.sidebar.write(
                        f"Attendance marked for Student {predicted_label}!")

        # Display the video frame
        video_placeholder.image(frame, channels="BGR", use_column_width=True)

    # Break the loop on "Stop" button press
    if st.button("Stop", key="stop_botton"):
        stop_flag = False

    # Release the video capture and close the app
    video_capture.release()


if __name__ == "__main__":
    main()
