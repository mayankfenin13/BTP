from typing import Optional
"""
Device utility for automatic device selection (MPS > CUDA > CPU)
"""
import torch

def get_device():
    """
    Returns the best available device for PyTorch.
    Priority: MPS (Apple Silicon) > CUDA (NVIDIA GPU) > CPU
    """
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def is_mps(device: Optional[str] = None) -> bool:
    if device is None:
        device = get_device()
    return device == "mps"

def get_num_workers(device: Optional[str] = None) -> int:
    """
    Return a safe default for DataLoader num_workers.
    MPS can be unstable with multiprocessing; prefer 0.
    """
    if is_mps(device):
        return 0
    return 2

def get_safe_infer_device() -> str:
    """
    For inference/evaluation paths that can crash on MPS, prefer CPU.
    """
    if is_mps():
        return "cpu"
    return get_device()

