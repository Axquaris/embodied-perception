import math
from typing import Union
import torch
from torch import nn, Tensor
import einops as eo
from jaxtyping import Float

from ember.math.homogenous import to_homogenous, from_homogenous


class Camera(nn.Module):
    def __init__(
        self,
        image_height: int,
        image_width: int,
        fov_x: float = math.pi / 2.0,  # 90 degrees
    ):
        super().__init__()

        # Extrinsics matrix: world coordinates to camera coordinates
        #   Places camera at (0, 0, -1) in world coordinates,
        #   because the extrinsics transform moves the world origin
        #   to (0, 0, 1) in camera coordinates
        self.register_buffer(   
            "extrinsics", torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                requires_grad=False,
            )
        )
        # Inverse extrinsics matrix: camera coordinates to world coordinates
        self.register_buffer(
            "inverse_extrinsics", torch.inverse(self.extrinsics)
        )
        
        # Intrinsics matrix: camera coordinates to image coordinates
        self.image_width = image_width
        self.image_height = image_height
        self.focal = 0.5 * image_width / math.tan(0.5 * fov_x)
        # Focal length calculations https://www.desmos.com/calculator/plcm03jtw3

        self.register_buffer(
            "intrinsics", torch.tensor(
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
            "inverse_intrinsics", torch.inverse(self.intrinsics)
        )

    def forward(
        self,
        points_world: Float[Tensor, "... 3"],
    ) -> Float[Tensor, "... 2"]:
        """Transforms 3D points to 2D image coordinates"""
        points_world_h = to_homogenous(points_world)
        
        # Transform points from world coordinates to camera coordinates
        points_camera_h = points_world_h @ self.extrinsics.T  # Left matrix-vector multiplication

        # Transform points from camera coordinates to image coordinates
        points_image_h = points_camera_h[..., :3] @ self.intrinsics.T  # Left matrix-vector multiplication

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
        xy_camera = torch.stack((x_image + 0.5, y_image + 0.5, h), dim=-1)
        
        xy_camera = xy_camera @ self.inverse_intrinsics.T
        
        points_camera = xy_camera * depths[..., None]
        points_camera[..., -1] *= -1  # Camera looks along -z axis

        # Create homogeneous coordinates
        points_camera_h = to_homogenous(points_camera)

        # Transform pixel coordinates to world coordinates
        points_world_h = points_camera_h @ self.inverse_extrinsics.T
        
        return from_homogenous(points_world_h)

