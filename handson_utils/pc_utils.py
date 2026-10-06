"""Point-cloud preprocessing helper, vendored from `utils/data.py` (repo root) so
that this tutorial folder is fully self-contained and doesn't need to reach
outside `tutorial/handson_cl/` to import anything at load time.
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
