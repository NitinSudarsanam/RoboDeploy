"""
Interoperability module: Zero-copy conversions between JAX, PyTorch, and NumPy.

The Zero-Copy flow:
1. Fetch: Observation starts as JAX (Sim) or NumPy (Real)
2. Convert: Real-world NumPy is moved to JAX via jnp.array(data)
3. Bridge: JAX is handed to PyTorch Policy via to_torch() (Zero-copy)
4. Command: Policy output (Torch) is moved to JAX for safety, then to NumPy for motors
"""

from typing import Union

import jax.numpy as jnp
import numpy as np


def to_torch(data: Union[jnp.ndarray, np.ndarray]) -> "torch.Tensor":
    """
    Convert JAX or NumPy array to PyTorch tensor (zero-copy when possible).
    
    Args:
        data: JAX array or NumPy array
        
    Returns:
        PyTorch tensor with shared memory (zero-copy on GPU)
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for to_torch(). Install with: pip install torch")

    if isinstance(data, jnp.ndarray):
        # JAX array -> NumPy (on CPU if needed)
        data = np.array(data)

    # NumPy -> PyTorch (zero-copy if possible)
    return torch.from_numpy(data)


def to_jax(data: Union["torch.Tensor", np.ndarray]) -> jnp.ndarray:
    """
    Convert PyTorch tensor or NumPy array to JAX array.
    
    Args:
        data: PyTorch tensor or NumPy array
        
    Returns:
        JAX array
    """
    if hasattr(data, "cpu"):  # PyTorch tensor
        data = data.detach().cpu().numpy()

    return jnp.array(data)


def to_numpy(data: Union[jnp.ndarray, "torch.Tensor"]) -> np.ndarray:
    """
    Convert JAX or PyTorch array to NumPy array.
    
    Args:
        data: JAX array or PyTorch tensor
        
    Returns:
        NumPy array
    """
    if isinstance(data, jnp.ndarray):
        return np.array(data)

    if hasattr(data, "cpu"):  # PyTorch tensor
        return data.detach().cpu().numpy()

    return np.array(data)
