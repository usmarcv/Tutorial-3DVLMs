"""Learnable CLIP-style temperature, vendored from `models/LogitScaleNetwork.py`
(repo root) so this tutorial folder has no import-time dependency on the rest
of the repo.
"""
import torch
import torch.nn as nn
import numpy as np


class LogitScaleNetwork(nn.Module):
    def __init__(self, init_scale=1 / 0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(init_scale))  # from OpenCLIP

    def forward(self, x=None):  # x accepted for interface parity, unused
        return self.logit_scale.exp().clamp(max=100)
