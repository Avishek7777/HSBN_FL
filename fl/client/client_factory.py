# fl/client/client_factory.py

import os
import importlib.util
import numpy as np
import torch.nn as nn


def load_random_head(
    arch_dir: str,
    in_dim: int,
    num_classes: int,
    seed: int | None = None,
) -> tuple[nn.Module, str]:
    """
    Randomly select an architecture file from arch_dir and instantiate
    its LocalHead class.

    Args:
        arch_dir    : path to the architectures/ folder
        in_dim      : HSBN z2 dim (32)
        num_classes : number of local task classes
        seed        : optional seed for reproducible arch selection

    Returns:
        (head module, arch filename)
    """
    files = [f for f in os.listdir(arch_dir) if f.endswith(".py")]
    if not files:
        raise RuntimeError(f"No .py architecture files found in {arch_dir}")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(files)

    spec = importlib.util.spec_from_file_location(
        "local_arch", os.path.join(arch_dir, chosen)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    head = module.LocalHead(in_dim=in_dim, num_classes=num_classes)
    return head, chosen