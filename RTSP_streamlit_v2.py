import cv2
import torch
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.write(f"Using device: {device}")

# Streamlit UI
st.title("Real-Time Object Detection with YOLOv8")

# Input field for RTSP URL
rtsp_url = st.text_input("Enter RTSP URL:", "")

# Load the pre-trained YOLOv8n model
model = YOLO('yolov8n.pt')  # Nano version of YOLOv8

# Session state to manage streaming
if 'streaming' not in st.session_state:
    st.session_state.streaming = False

# Button to toggle streaming
button_label = "Stop Streaming" if st.session_state.streaming else "Start Streaming"
if st.button(button_label) and rtsp_url:
    st.session_state.streaming = not st.session_state.streaming
    
    if st.session_state.streaming:
        cap = cv2.VideoCapture()
        cap.open(rtsp_url, cv2.CAP_FFMPEG)  # Explicitly use FFMPEG backend
        
        stframe = st.empty()  # Placeholder for video frames
        
        while cap.isOpened() and st.session_state.streaming:
            ret, frame = cap.read()
            if not ret:
                st.error("Streaming ended or failed to retrieve frame. Check your RTSP stream.")
                break

            # Perform inference
            results = model.predict(frame, conf=0.5, device=device)

            # Plot the detections
            for result in results:
                frame = result.plot()
            
            # Convert OpenCV frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            
            # Display the frame in Streamlit
            stframe.image(img, use_column_width=True)
        
        cap.release()
        st.session_state.streaming = False
