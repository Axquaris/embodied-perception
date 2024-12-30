import os
import pathlib as pth
import mediapy as media
import torch

from model import Gaussians3DConfig, Gaussians3D

if __name__ == "__main__":
    model = Gaussians3D(
        Gaussians3DConfig(
            num_points=10,
        )
    )

    with torch.no_grad():
        image = model()

        image_path = pth.Path(__file__).parent / "gaussians3d.png"
        media.write_image(
            image_path,
            image.cpu().numpy()
        )
    print(f"Image saved at: {str(image_path.absolute())}")
