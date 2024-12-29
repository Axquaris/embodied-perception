import torch
from torch import Tensor
from jaxtyping import Float


def to_homogenous(
    points: Float[Tensor, "..."]
) -> Float[Tensor, "..."]:
    """Converts points to homogenous coordinates by appending 1s."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)

def from_homogenous(
    points: Float[Tensor, "..."]
) -> Float[Tensor, "..."]:
    """Converts points from homogenous coordinates by dividing by the last element."""
    return points[..., :-1] / points[..., -1:]
