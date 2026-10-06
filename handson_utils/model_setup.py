"""Thin wrappers building the 3D encoder (Point-BERT) and the learnable
temperature (LogitScaleNetwork), fully vendored inside this tutorial folder
(see `point_bert.py`, `pointnet_util.py`, `logit_scale.py`) so this notebook
can run standalone in any environment - it only needs a handful of pip
packages (`torch`, `torch_redstone`, `einops`), never the rest of the
`3DMRL` repo.
"""
from .point_bert import build_point_bert
from .logit_scale import LogitScaleNetwork


def build_pointbert(scaling: int = 1, in_channel: int = 6, out_channel: int = 1280):
    """Build the trainable 3D encoder: Point-BERT (patchify + CLS-token
    transformer) + a linear projection into the shared 1280-dim OpenCLIP space.

    `scaling=1` is the smallest of the original repo's six size presets
    (`models/ppat.py`) - about 5M parameters, pure PyTorch (no custom CUDA
    kernels), fast enough to train live on CPU. No pretrained checkpoint exists
    for this preset, so the encoder starts randomly initialized - intentional,
    since this notebook's whole point is to watch contrastive training align it
    from scratch.
    """
    return build_point_bert(scaling=scaling, in_channel=in_channel, out_channel=out_channel)


def build_logit_scale(init_scale: float = 1 / 0.07):
    """The learnable temperature used by OpenCLIP/CLIP-style contrastive losses:
    logit_scale = exp(log_init).clamp(max=100). Same class the real trainer uses.
    """
    return LogitScaleNetwork(init_scale=init_scale)
