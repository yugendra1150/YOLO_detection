import cv2
import torch
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import asyncio
import time

# Ensure an asyncio event loop is set (helps avoid "no running event loop" issues)
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.write(f"Using device: {device}")

# Streamlit UI
st.title("Real-Time Object Detection with YOLOv8")

# Input field for RTSP URL
rtsp_url = st.text_input("Enter RTSP URL:", "")

# Load the pre-trained YOLOv8n model
model = YOLO('yolov8n.pt')  # Ensure this file is in your deployment environment

# Session state to manage streaming
if 'streaming' not in st.session_state:
    st.session_state.streaming = False

# Button to toggle streaming
button_label = "Stop Streaming" if st.session_state.streaming else "Start Streaming"
if st.button(button_label) and rtsp_url:
    st.session_state.streaming = not st.session_state.streaming

    if st.session_state.streaming:
        # Initialize video capture without explicitly specifying backend
        cap = cv2.VideoCapture(rtsp_url)
        stframe = st.empty()  # Placeholder for video frames

        while cap.isOpened() and st.session_state.streaming:
            ret, frame = cap.read()
            if not ret:
                st.error("Streaming ended or failed to retrieve frame. Check your RTSP stream.")
                break

            # Perform inference with YOLOv8
            results = model.predict(frame, conf=0.5, device=device)

            # Draw detections on the frame
            for result in results:
                frame = result.plot()

            # Convert BGR to RGB for display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)

            # Display the frame in Streamlit
            stframe.image(img, use_column_width=True)

            # Small delay to avoid hogging the CPU (use time.sleep in synchronous context)
            time.sleep(0.01)

        cap.release()
        st.session_state.streaming = False
