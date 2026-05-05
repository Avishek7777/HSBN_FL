# architectures/mlp_head.py

import torch.nn as nn


class LocalHead(nn.Module):
    """Lightweight MLP head. Suitable for low-resource clients."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, z):
        return self.net(z)