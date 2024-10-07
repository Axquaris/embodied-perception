"""
Execution:
    streamlit run "c:/Users/DOMCE/OneDrive/Desktop/Personal Projects/embodied-perception/scripts/webcam_to_streamlit_with_corners.py"
"""
import collections
import streamlit as st
import cv2
import torch
import kornia as K


def main():
    st.title("Webcam Live Feed")
    st.subheader("Corner Detection")

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Set the frame rate
    frame_rate = 60
    prev = 0

    # Create a 2x2 grid
    col1, col2 = st.columns(2)
    with col1:
        image_placeholder1 = st.empty()

    with col2:
        image_placeholder2 = st.empty()


    while cap.isOpened():
        time_elapsed = cv2.getTickCount() / cv2.getTickFrequency() - prev
        if time_elapsed > 1.0 / frame_rate:
            

            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert the frame to a PyTorch tensor (BCHW), B=1
            frame = K.utils.image_to_tensor(frame, keepdim=False).float() / 255.0

            if not ret:
                st.write("Failed to grab frame")
                break

            # Update the previous time
            prev = cv2.getTickCount() / cv2.getTickFrequency()

            # corner_map = K.feature.harris_response(frame_tensor, k=0.04)
            corner_map = K.feature.gftt_response(frame, grads_mode='diff')

            output_image = corner_map

            # Update the image placeholder with the new frame
            output_image /= output_image.abs().max()
            output_image = K.utils.tensor_to_image(output_image, keepdim=False)
            image_placeholder1.image(output_image, channels="RGB", clamp=True)

    cap.release()

if __name__ == "__main__":
    main()
