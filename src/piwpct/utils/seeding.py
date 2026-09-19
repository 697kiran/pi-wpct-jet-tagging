"""Deterministic seeding utility covering Python, NumPy, and PyTorch RNGs."""

import os
import random
from typing import Dict, Any
import numpy as np
import torch


def seed_everything(seed: int = 42) -> Dict[str, Any]:
    """Seed Python, NumPy, and PyTorch RNGs deterministically.

    Args:
        seed: The integer seed to apply.

    Returns:
        Dictionary with seeding metadata for logging.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return {
        "seed": seed,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_deterministic": True if torch.cuda.is_available() else False,
    }
