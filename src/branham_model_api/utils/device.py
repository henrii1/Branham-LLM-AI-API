"""
Device selection utilities.

As per Section 9.1: Device selection logic.
"""
import torch


def get_device(preference: str = "auto") -> torch.device:
    """
    Get the best available device based on preference.
    
    Args:
        preference: Device preference (mps, cuda, cpu, auto)
        
    Returns:
        torch.device: Selected device
        
    Priority (when auto):
    1. MPS if available and enabled (Apple Silicon)
    2. CUDA if available
    3. CPU as fallback
    """
    if preference == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS requested but not available")
    
    elif preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but not available")
    
    elif preference == "cpu":
        return torch.device("cpu")
    
    elif preference == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    else:
        raise ValueError(f"Unknown device preference: {preference}")


def get_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert string dtype to torch dtype.
    
    Args:
        dtype_str: String representation (fp16, bf16, fp32)
        
    Returns:
        torch.dtype: PyTorch data type
    """
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    
    if dtype_str not in dtype_map:
        raise ValueError(f"Unknown dtype: {dtype_str}. Must be one of {list(dtype_map.keys())}")
    
    return dtype_map[dtype_str]

