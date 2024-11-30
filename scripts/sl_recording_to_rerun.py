import time
import pyzed.sl as sl
import rerun as rr
import time
import pathlib as pth
from ember.util import PROJECT_ROOT

svo_file = str(PROJECT_ROOT / "data/zed_recordings/HD720_SN33087127_15-44-16.svo2")


def init_rerun():
    rr.init("vis_recording", spawn=True)
    rr.connect_tcp()  # Connect to a remote viewer

def main():
    init_rerun()

    zed_camera = sl.Camera()

    # Set configuration parameters
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_file)
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    # NONE	No depth map computation.
    # PERFORMANCE	Computation mode optimized for speed.
    # QUALITY	Computation mode designed for challenging areas with untextured surfaces.
    # ULTRA	Computation mode that favors edges and sharpness.
    # NEURAL	End to End Neural disparity estimation.
    # NEURAL_PLUS

    # Open the camera
    err = zed_camera.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(err))
        exit(-1)

    runtime_param = sl.RuntimeParameters()

    # Main loop grabbing images and depth values
    frame_idx = 0
    sl_image = sl.Mat()
    sl_depth = sl.Mat()
    sl_sensor_data = sl.SensorsData()

    while zed_camera.get_svo_position() < zed_camera.get_svo_number_of_frames() - 1:
        # Grab an image
        err = zed_camera.grab(runtime_param)
        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            print("End of SVO file reached.")
            break
        elif err == sl.ERROR_CODE.SUCCESS:
            rr.set_time_sequence("zed_camera/image_idx", frame_idx)

            # Get the timestamp
            timestamp = zed_camera.get_timestamp(sl.TIME_REFERENCE.IMAGE)
            rr.set_time_nanos("zed_camera/image_time", timestamp.get_nanoseconds())
            
            # Get the left image
            zed_camera.retrieve_image(sl_image, sl.VIEW.LEFT)  
            image = sl_image.numpy()
            image[..., :3] = image[..., 2::-1]  # BGRA -> RGBA

            rr.log("zed_camera/rgb", rr.Image(image))

            # Get the depth map
            zed_camera.retrieve_measure(sl_depth, sl.MEASURE.DEPTH)  
            depth = sl_depth.numpy()

            rr.log("zed_camera/depth", rr.DepthImage(depth))

            # Get imu data
            zed_camera.get_sensors_data(
                sl_sensor_data,
                time_reference=sl.TIME_REFERENCE.IMAGE
            )
            imu_data = sl_sensor_data.get_imu_data()
            print(imu_data.get_angular_velocity())
            print(imu_data.get_linear_acceleration())
            # rr.log("pose", rr.IMU(sl_imu))
            pose = imu_data.get_pose()
            rot_3x3 = pose.get_rotation_matrix().r
            translation = pose.get_translation().get()
            
            rr.log("zed_camera/pose", rr.Transform3D(
                mat3x3=rot_3x3,
                translation=translation
            ))
            
            rr.log("zed_camera/angular_velocity", rr.AnyValues(
                av=imu_data.get_angular_velocity()
            ))

            frame_idx = frame_idx + 1
            rr.reset_time()
        else:
            print("Error during grab: ", err)

    zed_camera.close()
    rr.disconnect()
    return 0


if __name__ == "__main__":
    main()
