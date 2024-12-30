from collections import defaultdict
import math
import time
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
import mediapy as media
import dataclasses
from gsplat import rasterization
from matplotlib import pyplot as plt


@dataclasses.dataclass
class Gaussians3DConfig:
    height: int = 256
    width: int = 256
    num_points: int = 10000

    # Gaussian random init params
    position_range: float = 1.4  # Cube "diameter"
    scale_max: float = 0.05

    # Prior params
    sparse_weight: float = 0.0
    slowness_weight: float = 0.0
    smoothness_weight: float = 0.0


class Gaussians3D(nn.Module):
    """Renderable gaussian 3D model."""

    POS_VIS_POINT_SIZE = 0.005

    def __init__(
        self,
        cfg: Gaussians3DConfig,
        gaussian_params=None,
    ):
        super().__init__()
        self.device = torch.device("cuda:0")
        self.cfg = cfg

        # Calculations for intrinsics matrix
        fov_x = math.pi / 2.0
        self.focal = 0.5 * float(self.cfg.width) / math.tan(0.5 * fov_x)

        # View transforms consructed for visible world coords to be in [-1, 1]
        self.world_to_camera = torch.eye(
            4, device=self.device, requires_grad=False,
        )[None]
        self.camera_to_image = torch.tensor(
            [
                [self.focal, 0, self.cfg.width / 2],
                [0, self.focal, self.cfg.height / 2],
                [0, 0, 1],
            ],
            device=self.device,
        )[None]

        self.background = torch.nn.Parameter(
            torch.tensor([0.0] * 3, device=self.device)[None]
        )

        if gaussian_params is None:
            self._init_gaussians()
        else:
            means, scales, quats, rgbs, opacities = gaussian_params

            self.latent = torch.nn.Parameter(
                torch.cat([means, scales, quats, rgbs, opacities], dim=-1)
            )

    def _init_gaussians(self):
        """Random gaussians"""
        position_range = self.cfg.position_range
        scale_max = self.cfg.scale_max

        means = position_range * (
            torch.rand(self.cfg.num_points, 3, device=self.device) - 0.5
        )

        # scales =  torch.exp(-5*torch.rand(self.cfg.num_points, 3, device=self.device))
        scales = torch.rand(self.cfg.num_points, 3, device=self.device)
        scales = scale_max * scales
        u = torch.rand(self.cfg.num_points, 1, device=self.device)
        v = torch.rand(self.cfg.num_points, 1, device=self.device)
        w = torch.rand(self.cfg.num_points, 1, device=self.device)
        quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
            ],
            -1,
        )

        rgbs = torch.ones(self.cfg.num_points, 3, device=self.device) * 0.5

        opacities = torch.zeros((self.cfg.num_points, 1), device=self.device)

        self.latent = torch.nn.Parameter(
            torch.cat([means, scales, quats, rgbs, opacities], dim=-1)
        )

    def get_gaussian_params(self, freeze_appearances=False):
        means, scales, quats, rgbs, opacities = torch.split(
            self.latent, [3, 3, 4, 3, 1], dim=-1
        )

        quats = F.normalize(quats, dim=-1)

        rgbs = rgbs # torch.sigmoid(2 * rgbs)
        opacities = torch.sigmoid(2 * opacities).flatten()

        if freeze_appearances:
            scales = scales.detach()
            quats = quats.detach()
            rgbs = rgbs.detach()
            opacities = opacities.detach()
        return means, scales, quats, rgbs, opacities

    def _render(
        self,
        gaussian_params,
        height=None,
        width=None,
    ):
        """Render the current scene."""
        means, scales, quats, rgbs, opacities = gaussian_params
        height = height or self.cfg.height
        width = width or self.cfg.width

        renders, _, _ = rasterization(
            means,
            quats,
            scales,
            opacities.flatten(),
            rgbs,
            self.world_to_camera,
            self.camera_to_image,
            width,
            height,
            packed=False,
            backgrounds=self.background,
        )
        return renders[0]

    def forward(
        self,
        height=None,
        width=None,
    ):
        """Renders the current scene."""
        gaussian_params = self.get_gaussian_params()
        return self._render(gaussian_params, height, width)

    def init_optimizer(self, stepsize, freeze_attributes=False):
        # if not hasattr(self, "optimizer"):
        param_list = [self.latent]
        if not freeze_attributes:
            param_list.append(self.background)

        self.optimizer = torch.optim.adam.Adam(
            param_list,
            stepsize,
            fused=True,
        )
        return self.optimizer

    def infer(
        self,
        target_poses: Tensor,  # (B, 4, 4)
        target_images: Tensor,  # (B, H, W, C)
        n_steps: int = 1000,
        stepsize: float = 0.01,
        freeze_attributes: bool = False,
        # Logging
        verbose: bool = False,
    ):
        """
        Fit the gaussians to the image.

        Note:
            - Currently inferring update for time t
            - previous_positions is pos at end of t-1 inference
            -
        """
        # Prior init
        enable_l1_loss = self.cfg.sparse_weight > 0 and not freeze_attributes

        # Inference init
        optimizer = self.init_optimizer(stepsize, freeze_attributes)

        # Logging init
        losses = defaultdict(list)
        recon_frames = []
        error_frames = []
        coeff_dot_images = []
        times = [0, 0]  # rasterization, backward

        for iter_idx in range(n_steps):
            # Raster Computation
            start = time.time()

            gaussian_params = self.get_gaussian_params(
                freeze_appearances=freeze_attributes
            )
            positions, scales, rotations, colors, opacities = gaussian_params
            recon_images = self._render(gaussian_params)

            torch.cuda.synchronize()
            times[0] += time.time() - start

            # Loss Computation
            loss = mse_loss = F.mse_loss(recon_images, target_images)

            if enable_l1_loss:
                l1_loss = torch.norm(opacities, p=1) * self.cfg.sparse_weight
                loss += l1_loss

            optimizer.zero_grad()

            # Backward Computation
            start = time.time()

            loss.backward()

            torch.cuda.synchronize()
            times[1] += time.time() - start

            optimizer.step()
            optimizer.zero_grad()

            # Logging
            #   This ensures CUDA garbage collection works
            with torch.no_grad():
                losses["loss"].append(loss.item())
                losses["mse_loss"].append(mse_loss.item())
                if enable_l1_loss:
                    losses["l1_loss"].append(l1_loss.item())

                if iter_idx % (n_steps // 30 + 1) == 0:
                    recon_frames.append(recon_images.squeeze().cpu().detach().numpy())

                    error_img = (recon_images - target_image) ** 2
                    error_img /= error_img.max()
                    error_frames.append(error_img.squeeze().cpu().detach().numpy())

                    scales = torch.ones_like(scales) * Gaussians3D.POS_VIS_POINT_SIZE
                    coeff_dot_image = self._render(
                        (positions, scales, rotations, colors, opacities),
                    )
                    coeff_dot_images.append(
                        coeff_dot_image.squeeze().detach().cpu().numpy()
                    )

                    if iter_idx % (n_steps // 10 + 1) == 0 and verbose:
                        # media.show_image(img)
                        print(
                            f"Iteration {iter_idx + 1}/{n_steps}, Loss: {loss.item()}"
                        )

            # Critical, clear computed gaussian params (from latents)
            #   Logging calcs need to be done with no_grad
            torch.cuda.empty_cache()

        return dict(
            losses=losses,
            recon_frames=recon_frames,
            error_frames=error_frames,
            coeff_dot_images=coeff_dot_images,
            total_render_time=times[0],
            total_backward_time=times[1],
            n_steps=n_steps,
        )

    def vis_inference(self, results: dict, fps=28):
        print(
            f"Totals(ms):\n"
            f"Rasterization: {results['total_render_time']*1000:.3f}, "
            f"Backward: {results['total_backward_time']*1000:.3f}"
        )
        render_time_per_step = results["total_render_time"] * 1000 / results["n_steps"]
        backward_time_per_step = (
            results["total_backward_time"] * 1000 / results["n_steps"]
        )
        print(
            f"Per step(ms):\n"
            f"Rasterization: {render_time_per_step:.5f}, "
            f"Backward: {backward_time_per_step:.5f}"
        )

        if len(results["recon_frames"]) > 0:
            media.show_videos(
                [
                    results["recon_frames"],
                    results["coeff_dot_images"],
                    results["error_frames"],
                ],
                ["Reconstruction", "Coefficients", "Error"],
                codec="gif",
                fps=fps,
                width=400,
            )

        fig, axs = plt.subplots(len(results["losses"]), 1, figsize=(8, 6))
        for i, (loss_name, loss_values) in enumerate(results["losses"].items()):
            axs[i].plot(loss_values)
            axs[i].set_title(f"{loss_name} over Inference Iterations")
            axs[i].set_xlabel("Inference Iteration")
            axs[i].set_ylabel(loss_name)
        plt.tight_layout()
        plt.show()
