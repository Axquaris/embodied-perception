import time
import pyzed.sl as sl
import rerun as rr
import time

def init_rerun():
    rr.init("vis_camera", spawn=True)
    rr.connect()  # Connect to a remote viewer

def main():
    init_rerun()

    # --- Initialize a Camera object and open the ZED
    # Create a ZED camera object
    zed = sl.Camera()

    # Set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = (
        sl.RESOLUTION.HD720
    )  # Use HD720 video mode for USB cameras
    # init_params.camera_resolution = sl.RESOLUTION.HD1200 # Use HD1200 video mode for GMSL cameras
    init_params.camera_fps = 60  # Set fps at 60

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(err))
        exit(-1)

    runtime_param = sl.RuntimeParameters()

    # --- Main loop grabbing images and depth values
    # Capture 50 frames and stop
    i = 0
    sl_image = sl.Mat()
    sl_depth = sl.Mat()
    while i < 60:
        # Grab an image
        if (
            zed.grab(runtime_param) == sl.ERROR_CODE.SUCCESS
        ):  # A new image is available if grab() returns SUCCESS
            # Log the image
            zed.retrieve_image(sl_image, sl.VIEW.LEFT)  # Get the left image
            image = sl_image.numpy()
            if err == sl.ERROR_CODE.SUCCESS:
                rr.log("rgb", rr.Image(image))
                
            else:
                print("Image ", i, " error:", err)

            # Display a pixel depth
            zed.retrieve_measure(sl_depth, sl.MEASURE.DEPTH)  # Get the depth map
            depth = sl_depth.numpy()
            if err == sl.ERROR_CODE.SUCCESS:
                rr.log("depth", rr.Image(depth))
            else:
                print("depth ", i, " error:", err)

            i = i + 1

            time.sleep(0.1)

    # --- Close the Camera
    zed.close()
    return 0


if __name__ == "__main__":
    main()
