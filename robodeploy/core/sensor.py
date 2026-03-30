"""
Sensor module: Base classes for robot sensors.
All camera sensors must provide both sim/ and real/ implementations.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters (pinhole camera model)."""

    focal_x: float  # Focal length in x (pixels)
    focal_y: float  # Focal length in y (pixels)
    principal_x: float  # Principal point x coordinate (pixels)
    principal_y: float  # Principal point y coordinate (pixels)
    width: int  # Image width in pixels
    height: int  # Image height in pixels


class BaseCamera(ABC):
    """
    Abstract base class for all camera sensors.
    
    To add a new sensor:
    1. Inherit from this class
    2. Provide a `sim/` implementation (MuJoCo renderer)
    3. Provide a `real/` implementation (Hardware driver)
    4. All camera outputs must be returned as jnp.ndarray (JAX Arrays)
    
    Sim implementations use JAX (MJX renders directly to GPU memory).
    Real implementations can use NumPy/OpenCV, but must output JAX arrays.
    """

    def __init__(self, intrinsics: CameraIntrinsics):
        """
        Initialize the camera.
        
        Args:
            intrinsics: Camera intrinsic parameters
        """
        self.intrinsics = intrinsics

    @abstractmethod
    async def get_rgb(self) -> jnp.ndarray:
        """
        Get RGB image from the camera.
        
        Returns:
            jnp.ndarray: Image of shape [H, W, 3] with uint8 values
        """
        raise NotImplementedError

    @abstractmethod
    async def get_depth(self) -> jnp.ndarray:
        """
        Get depth image from the camera.
        
        Returns:
            jnp.ndarray: Depth map of shape [H, W] with float32 values in meters
        """
        raise NotImplementedError

    async def get_rgbd(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get both RGB and depth images.
        
        Returns:
            Tuple of (rgb, depth) where:
            - rgb: [H, W, 3] uint8
            - depth: [H, W] float32 in meters
        """
        rgb, depth = await asyncio.gather(self.get_rgb(), self.get_depth())
        return rgb, depth
