from abc import ABC
import pyzed.sl as sl
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, Iterator
from pathlib import Path
import rerun as rr

from ember.util import PROJECT_ROOT, TEST_SVO2_PATH


@dataclass
class Frame:
    """Container for all data from a single ZED camera frame.

    Parameters
    ----------
    timestamp : int
        Frame timestamp in nanoseconds
    left_image : np.ndarray
        shape (H, W, RGBA)
    right_image : np.ndarray
        shape (H, W, RGBA)
    depth : np.ndarray
        shape (H, W)
    angular_velocity : np.ndarray
        Angular velocity from IMU
    linear_acceleration : np.ndarray
        Linear acceleration from IMU
    rotation : np.ndarray
        3x3 rotation matrix
    translation : np.ndarray
        3D translation vector
    frame_index : int
        Frame number in sequence
    """
    timestamp_ns: int  # nanoseconds
    image_left: np.ndarray  # RGBA image
    image_right: np.ndarray  # RGBA image
    depth: np.ndarray  # Depth map
    angular_velocity: np.ndarray  # Angular velocity from IMU
    linear_acceleration: np.ndarray  # Linear acceleration from IMU
    rotation: np.ndarray  # 3x3 rotation matrix
    translation: np.ndarray  # 3D translation vector
    frame_index: int  # Frame number in sequence
    name: Optional[str] = "zed_camera"  # Optional name for the frame

    def log_rerun(self):
        """Log frame data to Rerun."""
        rr.set_time_sequence(f"{self.name}/image_idx", self.frame_index)
        rr.set_time_nanos(f"{self.name}/image_time", self.timestamp_ns)
        
        rr.log(f"{self.name}/image_left", rr.Image(self.image_left))
        rr.log(f"{self.name}/image_right", rr.Image(self.image_right))
        rr.log(f"{self.name}/depth", rr.DepthImage(self.depth))
        rr.log(f"{self.name}/pose", rr.Transform3D(
            mat3x3=self.rotation,
            translation=self.translation
        ))
        rr.log(f"{self.name}/angular_velocity", rr.AnyValues(
            av=self.angular_velocity
        ))
        rr.reset_time()


class ZedInterface(ABC):
    """Interface for ZED camera implementations."""
    zed_camera: sl.Camera
    _frame_mats: Dict[str, sl.Mat | sl.SensorsData]

    def __iter__(self) -> Iterator[Frame]:
        return self
    
    def __next__(self) -> Frame:
        raise NotImplementedError
    
    def __enter__(self) -> 'ZedInterface':
        """Context manager entry. Returns self."""
        return self
    
    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[Any]
    ) -> None:
        """Context manager exit - ensures camera is closed. """
        self.zed_camera.close()
    
    def _get_timestamp_ns(self) -> int:
        return self.zed_camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
    
    def _get_images(self) -> tuple[np.ndarray, np.ndarray]:
        self.zed_camera.retrieve_image(self._frame_mats['left_image'], sl.VIEW.LEFT)
        self.zed_camera.retrieve_image(self._frame_mats['right_image'], sl.VIEW.RIGHT)

        left_image = self._frame_mats['left_image'].get_data().copy()
        left_image[..., :3] = left_image[..., 2::-1]

        right_image = self._frame_mats['right_image'].get_data().copy()
        right_image[..., :3] = right_image[..., 2::-1]

        return left_image, right_image
    
    def _get_depth(self) -> np.ndarray:
        self.zed_camera.retrieve_measure(self._frame_mats['depth'], sl.MEASURE.DEPTH)
        return self._frame_mats['depth'].get_data().copy()
    
    def _get_imu_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.zed_camera.get_sensors_data(self._frame_mats['sensors'], sl.TIME_REFERENCE.IMAGE)
        imu_data = self._frame_mats['sensors'].get_imu_data()
        
        angular_velocity = np.array(imu_data.get_angular_velocity())
        linear_acceleration = np.array(imu_data.get_linear_acceleration())

        pose = imu_data.get_pose()
        rotation = pose.get_rotation_matrix().r
        translation = pose.get_translation().get()

        return angular_velocity, linear_acceleration, rotation, translation


class ZedRecording(ZedInterface):
    """Wrapper for ZED camera recordings that provides easy iteration over frames."""
    
    def __init__(
        self,
        svo_path: str | Path = TEST_SVO2_PATH,
        depth_mode: sl.DEPTH_MODE = sl.DEPTH_MODE.ULTRA
    ) -> None:
        """Initialize ZED camera with an SVO recording.

        Parameters
        ----------
        svo_path : str | Path
            Path to .svo file (relative to PROJECT_ROOT)
        depth_mode : sl.DEPTH_MODE, optional
            ZED depth computation mode, by default sl.DEPTH_MODE.ULTRA

        Raises
        ------
        RuntimeError
            If camera fails to open
        """
        self.zed_camera = sl.Camera()
        
        # Configure initialization parameters
        init_params = sl.InitParameters()
        init_params.set_from_svo_file(str(PROJECT_ROOT / svo_path))
        init_params.depth_mode = depth_mode
        
        # Open camera
        err = self.zed_camera.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {err}")
            
        # Runtime parameters for grabbing frames
        self.runtime_params = sl.RuntimeParameters()
        
        # Preallocate sl.Mat objects for efficiency
        self._frame_mats = {
            'left_image': sl.Mat(),
            'right_image': sl.Mat(),
            'depth': sl.Mat(),
            'sensors': sl.SensorsData()
        }
        
        self._frame_idx = 0
    
    def __next__(self) -> Frame:
        """Get next frame from recording."""
        # Check if we've reached the end
        if self.zed_camera.get_svo_position() >= self.zed_camera.get_svo_number_of_frames() - 1:
            raise StopIteration
        
        # Grab next frame
        err = self.zed_camera.grab(self.runtime_params)
        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            raise StopIteration
        elif err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Error grabbing frame: {err}")
        
        timestamp_ns = self._get_timestamp_ns()
        left_image, right_image = self._get_images()
        depth = self._get_depth()
        angular_velocity, linear_acceleration, rotation, translation = self._get_imu_data()
        
        # Package everything into a Frame object
        frame = Frame(
            timestamp_ns=timestamp_ns,
            image_left=left_image,
            image_right=right_image,
            depth=depth,
            angular_velocity=angular_velocity,
            linear_acceleration=linear_acceleration,
            rotation=rotation,
            translation=translation,
            frame_index=self._frame_idx
        )
        
        self._frame_idx += 1
        return frame
    
    @property
    def frame_count(self) -> int:
        """Total number of frames in recording."""
        return self.zed_camera.get_svo_number_of_frames()
    
    @property
    def current_frame(self) -> int:
        """Current frame index."""
        return self.zed_camera.get_svo_position()
    
    def seek(self, frame_index: int) -> None:
        """Seek to specific frame index."""
        if not 0 <= frame_index < self.frame_count:
            raise ValueError(f"Frame index {frame_index} out of bounds [0, {self.frame_count})")
        self.zed_camera.set_svo_position(frame_index)
        self._frame_idx = frame_index


class ZedLive(ZedInterface):
    """Wrapper for ZED camera recordings that provides easy iteration over frames."""
    
    def __init__(
        self,
        depth_mode: sl.DEPTH_MODE = sl.DEPTH_MODE.ULTRA
    ) -> None:
        """Initialize ZED camera with an SVO recording.

        Parameters
        ----------
        depth_mode : sl.DEPTH_MODE, optional
            ZED depth computation mode, by default sl.DEPTH_MODE.ULTRA

        Raises
        ------
        RuntimeError
            If camera fails to open
        """
        self.zed_camera = sl.Camera()
        
        # Set configuration parameters
        init_params = sl.InitParameters()
        init_params.camera_resolution = (
            sl.RESOLUTION.HD720
        )
        # sl.RESOLUTION.HD720 video mode for USB cameras,
        # sl.RESOLUTION.HD1200 for GMSL cameras
        init_params.camera_fps = 60
        init_params.depth_mode = depth_mode
        
        # Open camera
        err = self.zed_camera.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {err}")
            
        # Runtime parameters for grabbing frames
        self.runtime_params = sl.RuntimeParameters()
        
        # Preallocate sl.Mat objects for efficiency
        self._frame_mats = {
            'left_image': sl.Mat(),
            'right_image': sl.Mat(),
            'depth': sl.Mat(),
            'sensors': sl.SensorsData()
        }
        
        self._frame_idx = 0
    
    def __next__(self) -> Frame:
        """Get next frame from recording."""
        err = self.zed_camera.grab(self.runtime_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Error grabbing frame: {err}")
        
        timestamp_ns = self._get_timestamp_ns()
        left_image, right_image = self._get_images()
        depth = self._get_depth()
        angular_velocity, linear_acceleration, rotation, translation = self._get_imu_data()
        
        # Package everything into a Frame object
        frame = Frame(
            timestamp_ns=timestamp_ns,
            image_left=left_image,
            image_right=right_image,
            depth=depth,
            angular_velocity=angular_velocity,
            linear_acceleration=linear_acceleration,
            rotation=rotation,
            translation=translation,
            frame_index=self._frame_idx
        )
        
        self._frame_idx += 1
        return frame


if __name__ == "__main__":
    with ZedRecording(TEST_SVO2_PATH) as zed_recording:
        for frame in zed_recording:
            print(frame.frame_index)
            print(frame.timestamp_ns)
            print(frame.image_left.shape)
            print(frame.image_right.shape)
            print(frame.depth.shape)
            print(frame.angular_velocity)
            print(frame.linear_acceleration)
            print(frame.rotation)
            print(frame.translation)
            break

    with ZedLive() as zed_live:
        for frame in zed_live:
            print(frame.frame_index)
            print(frame.timestamp_ns)
            print(frame.image_left.shape)
            print(frame.image_right.shape)
            print(frame.depth.shape)
            print(frame.angular_velocity)
            print(frame.linear_acceleration)
            print(frame.rotation)
            print(frame.translation)
            break
