# architectures/deep_mlp_head.py

import torch.nn as nn


class LocalHead(nn.Module):
    """Deeper MLP head for high-resource clients."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, z):
        return self.net(z)