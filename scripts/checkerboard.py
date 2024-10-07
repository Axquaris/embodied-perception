import time
import numpy as np
import torch
import pyzed.sl as sl
import rerun as rr
import time
import tyro
import kornia as K

from ember.corners import compute_corner_map


def init_rerun():
    rr.init("checkerboard.py", spawn=True)
    rr.connect()  # Connect to a remote viewer


def sl_image_to_np(sl_image):
    # BRGA to RGB
    image = sl_image.numpy()
    image = image[..., [2, 1, 0, 3]]
    return image
    

def main(recording_path: str = "C:/Users/DOMCE/OneDrive/Desktop/Personal Projects/embodied-perception/data/zed_recordings/HD720_SN33087127_15-44-16.svo2"):
    init_rerun()

    init_parameters = sl.InitParameters()
    init_parameters.set_from_svo_file(recording_path)

    # Open the camera
    zed_camera = sl.Camera()
    camera_status = zed_camera.open(init_parameters)
    if camera_status != sl.ERROR_CODE.SUCCESS: #Ensure the camera opened succesfully 
        print("Camera Open", camera_status, "Exit program.")
        exit(1)

    runtime_param = sl.RuntimeParameters()


    sl_image_left = sl.Mat()
    sl_image_right = sl.Mat()
    sl_depth = sl.Mat()
    while True:
        # If a new image is available ...
        if zed_camera.grab(runtime_param) == sl.ERROR_CODE.SUCCESS:
            # Retrieve frame data
            frame_idx = zed_camera.get_svo_position()
            # zed_camera.

            zed_camera.retrieve_image(sl_image_left, sl.VIEW.LEFT)
            zed_camera.retrieve_image(sl_image_right, sl.VIEW.RIGHT)
            zed_camera.retrieve_measure(sl_depth, sl.MEASURE.DEPTH)  # Get the depth map

            process_frame(
                frame_idx,
                image_left=sl_image_to_np(sl_image_left),
                image_right=sl_image_to_np(sl_image_right),
                depth=sl_depth.numpy(),
            )

        elif zed_camera.grab(runtime_param) == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            zed_camera.close()
            exit(-1)


def process_frame(frame_idx, image_left, image_right, depth):
    # Log frame data
    rr.set_time_sequence("frame_idx", frame_idx)
    rr.log("image_left", rr.Image(image_left))
    rr.log("image_right", rr.Image(image_right))
    rr.log("depth", rr.Image(depth))

    for name, frame in [("image_left", image_left), ("image_right", image_right)]:
        # Convert to tensor
        frame = K.utils.image_to_tensor(frame, keepdim=False).float() / 255.0

        corner_map = compute_corner_map(frame)
        rr.log(f"{name}_corners", rr.Image(
            K.utils.tensor_to_image(corner_map, keepdim=False)
        ))



if __name__ == "__main__":
    tyro.cli(main)
