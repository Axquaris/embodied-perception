import os
import pathlib as pth
import mediapy as media
import torch
import rerun as rr

import pyzed.sl as sl
import pyzed as zed
import viser

from ember.math.camera import Camera
from ember.math.quat import qvec2rotmat
from ember.wrappers.zed import ZedRecording
from ember.wrappers.viser import ViserServer

from model import Gaussians3DConfig, Gaussians3D


def write_to_rerun(
    image, depth, means, rgbs, point_cloud_xyzrgba
):
    rr.init("unproject", spawn=True)
    rr.connect_tcp()  # Connect to a remote viewer

    # Thumb is +X, Index is +Y, Middle is +Z
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    rr.log(f"image_left", rr.Image(image))
    rr.log(f"depth", rr.DepthImage(depth))
    rr.log(f"points_world", rr.Points3D(means.cpu(), colors=rgbs.cpu()))

    point_cloud_xyzrgba = point_cloud_xyzrgba.reshape(-1, 4)
    print(point_cloud_xyzrgba.shape)
    rr.log("xyzrgba", rr.Points3D(point_cloud_xyzrgba[..., :3]))


def write_to_image(
    image
):
    image_path = pth.Path(__file__).parent / "gaussians3d.png"
    media.write_image(
        image_path,
        image.cpu().numpy()
    )
    print(f"Image saved at: {str(image_path.absolute())}")


if __name__ == "__main__":
    # Fetch rgbd image from zed recording
    svo_path = "data/zed_recordings/HD720_SN33087127_15-44-16.svo2"

    with ZedRecording(svo_path) as zed:
        frame, point_cloud_xyzrgba = next(zed)
    
    # Unproject the depth to 3D points
    image = torch.as_tensor(frame.image_left, device="cuda")
    depth = torch.as_tensor(frame.depth, device="cuda")  # Convert to meters
    invalid_mask = torch.isnan(depth)

    viser_camera = Camera(
        image_height=image.shape[0],
        image_width=image.shape[1],
    ).to("cuda")

    points_world = viser_camera.unproject(depth) #/ 1000.  # Convert to meters
    
    means = points_world[~invalid_mask].reshape(-1, 3)
    rgbs = image[..., :3][~invalid_mask].reshape(-1, 3) / 255.
    n_points = means.shape[0]

    # Initialize gaussian 3d model with colored 3d points,
    #   and appropriate gaussian size
    model = Gaussians3D(
        Gaussians3DConfig(
            num_points=n_points,
        ),
        gaussian_params=(
            means,
            torch.ones(n_points, 3, device="cuda") * 1. / 1000,  # in meters
            torch.ones(n_points, 4, device="cuda"),
            rgbs,
            torch.ones(n_points, 1, device="cuda"),
        )
    )

    # Render an image for debugging
    with torch.no_grad():
        image = model()
    
    # write_to_image(image)

    write_to_rerun(
        image, depth, means, rgbs, point_cloud_xyzrgba
    )

    # Live visualization via viser
    server = ViserServer()

    # Bind the UI to the camera
    viser_client, viser_camera = server.wait_for_camera()

    def on_camera_update(viser_camera: viser.CameraHandle):
        camera_to_world = torch.eye(4, device="cuda")
        
        camera_to_world[ :3, :3] = qvec2rotmat(
            viser_camera.wxyz
        ).to("cuda")
        camera_to_world[:3, 3] = torch.tensor(
            viser_camera.position, device="cuda"
        )

        model.world_to_camera = camera_to_world.inverse()[None]

    viser_camera.on_update(on_camera_update)
    
    # #######################
    # # Main rendering loop #
    # #######################
    while True:
        with torch.no_grad():
            image = model()

        viser_client.scene.set_background_image(
            image=image.cpu().numpy(),
            # depth=depth.cpu().numpy(),
            # format="jpeg",
            # jpeg_quality=70,
        )
