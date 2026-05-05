# architectures/linear_head.py

import torch.nn as nn


class LocalHead(nn.Module):
    """Single linear projection. Minimal compute, strong regularization."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, z):
        return self.fc(z)