import torch
import pytest
from ember.math.camera import Camera

@pytest.fixture
def camera():
    image_height = 480
    image_width = 640
    # Create a Camera instance with realistic default values for intrinsics and extrinsics
    return Camera(image_height, image_width)

def test_initialization(camera):
    # Check dimensions
    assert camera.image_height == 480
    assert camera.image_width == 640
    
    # Check attributes
    assert hasattr(camera, 'extrinsics')
    assert hasattr(camera, 'inverse_extrinsics')
    assert hasattr(camera, 'intrinsics')
    assert hasattr(camera, 'inverse_intrinsics')

    # Check if attributes are tensors of correct shape
    assert camera.intrinsics.shape == (3, 3)
    assert camera.extrinsics.shape == (4, 4)
    assert camera.inverse_intrinsics.shape == (3, 3)
    assert camera.inverse_extrinsics.shape == (4, 4)

def test_forward(camera):
    # Simulate a point in world coordinates
    points = torch.tensor([[0.0, 0.0, 1.0]])
    
    # Transform to image coordinates
    image_points = camera.forward(points)
    
    # Validate output shape
    assert image_points.shape == (1, 2)

    # Validate output values
    expected_x = camera.image_width / 2
    expected_y = camera.image_height / 2
    assert pytest.approx(image_points[0, 0].item(), 0.00001) == expected_x
    assert pytest.approx(image_points[0, 1].item(), 0.00001) == expected_y

def test_unproject(camera):
    # Create a depth map
    depths = torch.ones((camera.image_height, camera.image_width))

    # Transform to world coordinates
    world_coords = camera.unproject(depths)

    # Validate output shape
    assert world_coords.shape == (camera.image_height, camera.image_width, 3)

    # Validate depth correspondence (negative z-direction assumed)
    assert torch.allclose(world_coords[..., 2], -depths - 1, atol=1e-5)
