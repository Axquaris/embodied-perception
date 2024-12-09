from typing import Tuple

import numpy as np
import time
import torch
from einops import einsum

from ember.wrappers.viser import ViserServer
from ember.torch_perf_utils import Timer
from ember.math.quat import qvec2rotmat


#######################
# Rendering Functions #
#######################
@torch.jit.script
def compute_xyo(H: int, W: int, aspect_ratio: float) -> torch.Tensor:
    x = torch.linspace(-1, 1, W, device='cuda')
    y = torch.linspace(-1, 1, H, device='cuda') / aspect_ratio
    y, x = torch.meshgrid(y, x)
    o = torch.ones_like(x)
    xyo = torch.stack([x, y, o], dim=-1)
    return xyo


@torch.jit.script
def sphere_sdf(
    points: torch.Tensor,
    sphere_pos: torch.Tensor,
    sphere_radius: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the signed distance function (SDF) for a sphere.

    Parameters
    ----------
    points : torch.Tensor
        Points in space to calculate the SDF for, shape (N, 3).

    Returns
    -------
    torch.Tensor
        Signed distance from each point to the sphere surface, shape (N,).
    """
    return torch.norm(points - sphere_pos, dim=-1) - sphere_radius


@torch.jit.script
def sphere_normal(points: torch.Tensor, sphere_pos, sphere_radius) -> torch.Tensor:
    """
    Calculate the normal vectors for a sphere at given points.

    Parameters
    ----------
    points : torch.Tensor
        Points in space to calculate the normals for, shape (N, 3).

    Returns
    -------
    torch.Tensor
        Normal vectors at each point, shape (N, 3).
    """
    eps = 1e-8
    sdf = sphere_sdf(points, sphere_pos, sphere_radius)
    sdf_dx = sphere_sdf(
        points + torch.tensor([eps, 0., 0.], device="cuda"),
        sphere_pos, sphere_radius
    ) - sdf
    sdf_dy = sphere_sdf(
        points + torch.tensor([0., eps, 0.], device="cuda"),
        sphere_pos, sphere_radius
    ) - sdf
    sdf_dz = sphere_sdf(
        points + torch.tensor([0., 0., eps], device="cuda"),
        sphere_pos, sphere_radius
    ) - sdf

    return torch.nn.functional.normalize(torch.stack([sdf_dx, sdf_dy, sdf_dz], dim=-1), dim=-1)


@torch.jit.script
def sphere_trace(
    rays: torch.Tensor,
    sphere_pos, sphere_radius,
    camera_center: torch.Tensor = torch.tensor([0, 0, 0], device="cuda"),
    max_steps: int = 20,
    min_dist: float = 0.001,
    max_dist: float = 100.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform sphere tracing to find the intersection points of rays with a sphere.

    Parameters
    ----------
    rays : torch.Tensor
        Rays through camera pixels in world coordinates, shape (N, 3).
    camera_center : torch.Tensor, optional
        The center of the camera, shape (3,). Default is [0, 0, 0]
    max_steps : int, optional
        Maximum number of tracing steps. Default is 20.
    min_dist : float, optional
        Minimum distance to consider an intersection. Default is 0.001.
    max_dist : float, optional
        Maximum distance to trace. Default is 100.0.

    Returns
    -------
    torch.Tensor
        Depths of intersection points, shape (N, 1).
    """
    points = torch.zeros_like(rays, device="cuda")
    depths = points[..., :1]

    for i in range(max_steps):
        points = camera_center + depths * rays
        dist = sphere_sdf(points, sphere_pos, sphere_radius).unsqueeze(-1)
        depths += dist
        if ((dist < min_dist) | (depths > max_dist)).all():
            break

    return depths, points


if __name__ == "__main__":
    server = ViserServer()

    position_slider = server.gui.add_vector3(
        "Position",
        initial_value=(0, 0, 0),
        min=(-5, -5, -5),
        max=(5, 5, 5),
    )
    radius_slider = server.gui.add_slider(
        "Radius",
        min=0,
        max=5,
        step=0.1,
        initial_value=1
    )

    resolution_slider = server.gui.add_slider(
        "Resolution", min=64, max=1024, step=64, initial_value=256
    )
    fov_slider = server.gui.add_slider(
        "FOV", min=30, max=170, step=10, initial_value=60
    )
    trace_iter_slider = server.gui.add_slider(
        "Trace Iters", min=1, max=200, step=1, initial_value=20
    )


    # Bind the UI to the camera
    client, camera = server.wait_for_camera()

    camera_updated = True
    @camera.on_update
    def _(_):
        global camera_updated
        camera_updated = True

    @fov_slider.on_update
    def _(_) -> None:
        camera.fov = fov_slider.value * np.pi / 180

    # Camera parameter fetchers
    def get_updated_camera_intrinsics():
        W = resolution_slider.value
        H = int(W / camera.aspect)
        focal_x = W / 2 / np.tan(camera.fov/2)
        focal_y = H / 2 / np.tan(camera.fov/2)

        return W, H, focal_x, focal_y


    def get_updated_camera_extrinsics():
        rot_c2w = torch.tensor(camera.wxyz).view(1, 4)
        
        return rot_c2w
    
    #######################
    # Main rendering loop #
    #######################
    frame_idx = 0
    last_intrinsics = None
    last_extrinsics = None

    while True:
        intrinsics = get_updated_camera_intrinsics()
        if intrinsics != last_intrinsics:
            W, H, focal_x, focal_y = intrinsics

            xyo = compute_xyo(H, W, camera.aspect)

            camera_updated = True
            last_intrinsics = intrinsics

        rot_c2w = get_updated_camera_extrinsics()
        if last_extrinsics is None or (rot_c2w != last_extrinsics).any():
            camera_center = torch.tensor(camera.position, device="cuda")
            rot_c2w_mat = qvec2rotmat(camera.wxyz).to("cuda")

            pixels__world_rot = einsum(
                xyo, rot_c2w_mat, '... p, o p-> ... o'
            )
            pixels__world = pixels__world_rot + camera_center
            rays = torch.nn.functional.normalize(pixels__world_rot, dim=-1)

            camera_updated = True
            last_extrinsics = rot_c2w

        if camera_updated:
            camera_updated = False

            with Timer("render"):
                sphere_pos = torch.tensor(position_slider.value, device="cuda")
                sphere_radius = torch.tensor(radius_slider.value, device="cuda")

                depth, intersections_world = sphere_trace(
                    rays,
                    camera_center=camera_center,
                    max_steps=trace_iter_slider.value,
                    sphere_pos=sphere_pos,
                    sphere_radius=sphere_radius,
                )

                image = sphere_normals = sphere_normal(
                    intersections_world,
                    sphere_pos, sphere_radius
                )
                image[depth.squeeze() >= 9] = 0.  # Set background to 0 value

                image -= image.min()
                image /= image.max()

            with Timer("set_bg_image"):
                client.scene.set_background_image(
                    image=image.cpu().numpy(),
                    depth=depth.cpu().numpy(),
                    # format="jpeg",
                    # jpeg_quality=70,
                )
        
            if frame_idx % 100 == 0:
                print(f"Frame {frame_idx}")
                Timer.show_recorder()

            frame_idx += 1