"""
Camera module for common imaging, and projective geometry operations.

"Standard" Conventions:
- World coordinates:
    Origin: Center of the world, usually first camera position
    Right: +X
    Up: +Y
    Forward: +Z
- Camera coordinates:
    Origin: Camera center
    Right: +X
    Up: -Y
    Forward: +Z
- Image coordinates:
    Origin: Bottom left corner
    Right: +X
    Up: +Y
    Forward: +Z
"""
import math
import torch
from torch import nn, Tensor
from enum import Enum
from jaxtyping import Float
import viser

from ember.math.homogenous import to_homogenous, from_homogenous
from ember.math.quat import qvec2rotmat



class CameraConvention(Enum):
    """
    Enum for camera coordinate conventions.

    Expressed as Right, Up, Forward
    """
    STANDARD = "+X +Y +Z"
    GSPLAT = dict(
        world="+Z forward, +X right, +Y up",
        camera="+Z forward, +X right, +Y down",
        image="+Y down, +X right",
    )
    
    VISER = COLMAP = OPENCV = "+X -Y +Z"
    NERFSTUDIO = OPENGL = BLENDER = "+X +Y -Z"


class Camera(nn.Module):
    camera_to_world: Float[Tensor, "4 4"]
    world_to_camera: Float[Tensor, "4 4"]
    camera_to_image: Float[Tensor, "3 3"]
    image_to_camera: Float[Tensor, "3 3"]

    def __init__(
        self,
        image_height: int,
        image_width: int,
    ):
        super().__init__()

        # Extrinsics matrix: world coordinates to camera coordinates
        #   Places camera at (0, 0, -1) in world coordinates,
        #   because the extrinsics transform moves the world origin
        #   to (0, 0, 1) in camera coordinates
        self.register_buffer(   
            "world_to_camera", torch.eye(4, requires_grad=False)
        )
        # Inverse extrinsics matrix: camera coordinates to world coordinates
        self.register_buffer(
            "camera_to_world", self.world_to_camera
        )
        
        # Intrinsics matrix: camera coordinates to image coordinates
        self.image_width = image_width
        self.image_height = image_height
        fov_x = math.pi / 2.0  # 90 degrees
        self.focal = 0.5 * image_width / math.tan(0.5 * fov_x)
        # Focal length calculations https://www.desmos.com/calculator/plcm03jtw3

        self.register_buffer(
            "camera_to_image", torch.tensor(
                [
                    [self.focal, 0, image_width / 2.],
                    [0, self.focal, image_height / 2.],
                    [0, 0, 1],
                ],
                requires_grad=False,
            )
        )
        # Inverse intrinsics matrix: image coordinates to camera coordinates
        self.register_buffer(
            "image_to_camera", torch.inverse(self.camera_to_image)
        )

    # @staticmethod
    # def from_viser_camera(
    #     viser_camera: viser.CameraHandle,
    # ) -> "Camera":
    #     """Create a Camera instance from a Viser camera handle"""
    #     fov_y = viser_camera.fov
    #     fov_x = viser_camera.aspect * fov_y
    #     viser_camera.client.
    #     viser_camera.wxyz
    #     viser_camera.position

    #     camera = Camera(
    #         image_height=1,
    #         image_width=viser_camera.width,
    #     )
    #     camera.extrinsics[:3, :3] = viser_camera.rotation

    #     camera.extrinsics = viser_camera.extrinsics
    #     camera.inverse_extrinsics = viser_camera.inverse_extrinsics
    #     camera.intrinsics = viser_camera.intrinsics
    #     camera.inverse_intrinsics = viser_camera.inverse_intrinsics

    #     return camera

    def forward(
        self,
        points_world: Float[Tensor, "... 3"],
    ) -> Float[Tensor, "... 2"]:
        """Transforms 3D points to 2D image coordinates"""
        points_world_h = to_homogenous(points_world)
        
        # Transform points from world coordinates to camera coordinates
        points_camera_h = points_world_h @ self.world_to_camera.T  # Left matrix-vector multiplication

        # Transform points from camera coordinates to image coordinates
        points_image_h = points_camera_h[..., :3] @ self.camera_to_image.T  # Left matrix-vector multiplication

        return from_homogenous(points_image_h)  # Return only x, y coordinate components

    def unproject(
        self,
        depths: Float[Tensor, "..."],
    ) -> Float[Tensor, "... 3"]:
        """Unproject a depth map to world coordinates using camera intrinsics and extrinsics."""
        # Create grid of pixel coordinates in image space
        y_image, x_image = torch.meshgrid(
            torch.arange(self.image_height), torch.arange(self.image_width), indexing="ij"
        )
        y_image = y_image.flip(0)  # Origin is in bottom left corner

        # Convert pixel coordinates to camera coordinates
        #  Adding 0.5 to x_image and y_image to get the center of the pixel
        h = torch.ones_like(x_image)
        xyh_image = torch.stack((x_image + 0.5, y_image + 0.5, h), dim=-1).to(
            dtype=depths.dtype, device=depths.device
        )
        
        xyh_camera = xyh_image @ self.image_to_camera.T

        points_camera = xyh_camera * depths[..., None]
        # points_camera[..., -1] *= -1  # Camera looks along -z axis

        # Create homogeneous coordinates
        points_camera_h = to_homogenous(points_camera)

        # Transform pixel coordinates to world coordinates
        points_world_h = points_camera_h @ self.camera_to_world.T
        
        return from_homogenous(points_world_h)

