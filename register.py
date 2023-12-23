import streamlit as st
import cv2
import mediapipe as mp
import pandas as pd
import sqlite3
from sklearn.svm import SVC
import os
import time


# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Database Initialization
DB_PATH = "attendance.db"

current_directory = os.getcwd()


def initialize_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            ID TEXT PRIMARY KEY,
            Name TEXT NOT NULL,
            Attendance TEXT NOT NULL
        );
    ''')
    conn.commit()
    conn.close()


initialize_database()

# SVC Model
model = SVC()


def save_frames(frames, id_inp, folder_path="dataset"):
    print("current directory:", current_directory)
    cold_path = os.path.join(current_directory, folder_path)
    print("cold path:", cold_path)
    folder_path = os.path.join(cold_path, f"{id_inp}")

    print("folder path:", folder_path)

    # Create the new directory
    os.makedirs(folder_path, exist_ok=True)

    for idx, frame in enumerate(frames):
        filename = os.path.join(folder_path, f"frame_{idx}.jpg")
        cv2.imwrite(filename, frame)


@st.cache(allow_output_mutation=True)
def load_model():
    return model


def main():
    st.title("Student Attendance System")

    if "time" not in st.session_state:
        st.session_state.time = time.time()  # import time at the top

# ---------------Get student details--------------------------------------
    id_input = st.text_input("Enter Student ID:")
    name_input = st.text_input("Enter Student Name:")


# ----------------Adding Student Information to Database-------------------
    if st.button("Submit Student Information", key="submit_botton"):
        if id_input and name_input:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO attendance (ID, Name, Attendance) VALUES (?, ?, ?)",
                           (id_input, name_input, "Absent"))
            conn.commit()
            conn.close()
            st.success("Student information submitted successfully!")

    # Record video
    recording = False
    frames = []

# ------------------ Recording the Video and saving frames as dataset-----------------
    if st.button("Start Recording", key="start_botton"):
        recording = True
        start_time = time.time()  # Get current time
        print("start time:", start_time)

        # Display the video feed in a specified region
        video_placeholder = st.empty()

        cap = cv2.VideoCapture(0)
        while recording:
            current_time = time.time()  # Get current time
            elapsed_time = current_time - start_time
            print("elapsed time", elapsed_time)

            if elapsed_time >= 5:  # Stop recording after 5 seconds
                recording = False

            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                #  Convert the frame from BGR to RGB for display in Streamlit
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(
                    rgb_frame, channels="RGB", use_column_width=True)
                # if st.button("Stop Recording", key="stop_botton"):
                #     recording = False
        cap.release()
        cv2.destroyAllWindows()

        # Save frames
        save_frames(frames, id_input)
        st.success("Video recording completed and frames saved!")

# -------------------- Image Augmentation and Dataset Formation -----------------------
    if st.button("Process Video", key="process_botton"):
        # video data augmentation part --------
        print("frame data augmentation")


# -------------------- Model Training --------------------------------------
    if st.button("Register", key="register_botton"):
        # ---------------- Extract facial landmarks ----------------------
        with mp_face_mesh.FaceMesh() as face_mesh:
            landmarks = []
            for frame in frames:
                results = face_mesh.process(frame)
                if results.multi_face_landmarks:
                    for landmark in results.multi_face_landmarks[0].landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])

        # --------------- Train SVC model ---------------------------------
        if id_input and name_input and landmarks:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            # Add or update attendance record in database
            cursor.execute("INSERT OR REPLACE INTO attendance (ID, Name, Attendance) VALUES (?, ?, ?)",
                           (id_input, name_input, "Absent"))
            conn.commit()
            conn.close()

            # Train SVC model
            X_train = [landmarks]
            y_train = [name_input]
            model.fit(X_train, y_train)

    # ------------------------- Display attendance records -------------------
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM attendance", conn)
    conn.close()

    st.write(df)


if __name__ == "__main__":
    main()
