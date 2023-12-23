import streamlit as st
import cv2


def main():
    st.title("Live Video Feed in Streamlit")

    # Display the video feed in a specified region
    video_placeholder = st.empty()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if ret:
            # Convert the frame from BGR to RGB for display in Streamlit
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(
                rgb_frame, channels="RGB", use_column_width=True)


if __name__ == "__main__":
    main()
