"""Point-cloud preprocessing helper, vendored from the original 3DMRL research
code (`utils/data.py`) so that this repository is fully self-contained.
"""
import numpy as np


def normalize_pc(pc: np.ndarray) -> np.ndarray:
    """Mean-center a point cloud and rescale it to fit inside the unit sphere."""
    pc = pc - np.mean(pc, axis=0)
    if np.max(np.linalg.norm(pc, axis=1)) < 1e-6:
        pc = np.zeros_like(pc)
    else:
        pc = pc / np.max(np.linalg.norm(pc, axis=1))
    return pc
