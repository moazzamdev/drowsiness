import streamlit as st
import torch
import numpy as np
import cv2
import time
import posixpath
import pathlib
# Suppressing torch.hub download progress
import logging
logging.getLogger("transformers.file_utils").setLevel(logging.WARNING)
# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path="last.pt", force_reload=True)
model.conf = 0.55
model.iou = 0.4
model.classes = [4, 5]
FRAME_WINDOW = st.image([])
# Function to perform object detection
def detect_objects(frame):
    # Perform object detection
    results = model(frame, size=320)
    return results

# Main function to run the Streamlit app
def main():
    st.title("YOLOv5 Object Detection")

    # Open camera and perform object detection
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Unable to open camera.")
        return

    # Continuous detection loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Failed to capture image from camera.")
            break

        # Convert the frame to RGB (OpenCV uses BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform object detection
        results = detect_objects(frame_rgb)

        # Render the detection results on the frame
        output_frame = np.squeeze(results.render())
        FRAME_WINDOW.image(output_frame)




    cap.release()


if __name__ == "__main__":
    main()
