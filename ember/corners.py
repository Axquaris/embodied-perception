"""
Execution:
    streamlit run "c:/Users/DOMCE/OneDrive/Desktop/Personal Projects/embodied-perception/scripts/webcam_to_streamlit_with_corners.py"
"""
import collections
import streamlit as st
import cv2
import torch
import kornia as K


keynet = K.feature.KeyNet(pretrained=True)


def compute_corner_map(image, k=0.04):

    # corner_map = K.feature.harris_response(image, k=k)
    # corner_map = K.feature.gftt_response(image, grads_mode='diff')
    corner_map = keynet(image)


    # Update the image placeholder with the new frame
    # corner_map /= corner_map.abs().max()
    # corner_map = K.utils.tensor_to_image(corner_map, keepdim=False)
    return corner_map

