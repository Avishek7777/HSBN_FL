# fl/runner.py

import os
import copy
import numpy as np
import torch
from typing import Dict, List

from hsbn.network.hsbn import HSBN, build_hsbn
from fl.client.base_client import FLClient
from fl.client.client_factory import load_random_head
from fl.server.aggregator import fedavg
from fl.data.dirichlet import dirichlet_partition
from fl.data.loaders import build_client_loaders, build_global_test_loader


class FLRunner:
    """
    Orchestrates the full FL training loop.

    Each round:
        1. Sample a fraction of clients
        2. Distribute current global HSBN state to each
        3. Each client trains locally
        4. Aggregate HSBN weights via FedAvg
        5. Evaluate on global test set
    """

    def __init__(self, cfg: dict, device: str = "cpu"):
        self.cfg    = cfg
        self.device = device

        fl_cfg   = cfg["fl"]
        data_cfg = cfg["data"]

        self.num_clients     = fl_cfg["num_clients"]
        self.num_rounds      = fl_cfg["num_rounds"]
        self.client_fraction = fl_cfg["client_fraction"]
        self.local_epochs    = fl_cfg["local_epochs"]
        self.arch_dir        = fl_cfg["arch_dir"]
        self.data_root       = data_cfg["root"]
        self.batch_size      = data_cfg["batch_size"]
        self.alpha           = data_cfg["dirichlet_alpha"]
        self.seed            = fl_cfg.get("seed", 42)

        # Build global HSBN
        self.global_hsbn = build_hsbn(cfg).to(device)

        # Partition data
        from torchvision.datasets import CIFAR100
        import torchvision.transforms as T
        _ds = CIFAR100(root=self.data_root, train=True, download=True,
                       transform=T.ToTensor())
        targets = np.array(_ds.targets)
        partition = dirichlet_partition(
            targets, self.num_clients, self.alpha, seed=self.seed
        )

        # Build per-client loaders
        self.client_loaders = build_client_loaders(
            partition, self.data_root, self.batch_size
        )
        self.client_sizes = {
            cid: len(idxs) for cid, idxs in partition.items()
        }

        # Build client objects (fixed arch per client across all rounds)
        self.clients: Dict[int, FLClient] = {}
        for cid in range(self.num_clients):
            head, arch_name = load_random_head(
                arch_dir    = self.arch_dir,
                in_dim      = self.global_hsbn.d2,   # 32
                num_classes = data_cfg["num_fine_classes"],
                seed        = self.seed + cid,        # deterministic per client
            )
            client_hsbn = copy.deepcopy(self.global_hsbn)
            self.clients[cid] = FLClient(
                client_id    = cid,
                hsbn         = client_hsbn,
                local_head   = head,
                dataloader   = self.client_loaders[cid],
                arch_name    = arch_name,
                num_classes  = data_cfg["num_fine_classes"],
                lr           = fl_cfg["lr"],
                local_epochs = self.local_epochs,
                device       = device,
            )

        # Global test loader
        self.test_loader = build_global_test_loader(self.data_root, self.batch_size)

        self.history: List[dict] = []

    def run(self):
        rng = np.random.default_rng(self.seed)

        for round_idx in range(self.num_rounds):
            # Sample clients
            n_selected = max(1, int(self.num_clients * self.client_fraction))
            selected   = rng.choice(self.num_clients, size=n_selected, replace=False)

            round_metrics = []
            client_states  = []
            client_weights = []

            for cid in selected:
                client = self.clients[cid]
                # Push global HSBN state
                client.load_hsbn_state(
                    copy.deepcopy(self.global_hsbn.state_dict())
                )
                # Local training
                metrics = client.train_round(global_epoch=round_idx)
                round_metrics.append(metrics)
                client_states.append(client.get_hsbn_state())
                client_weights.append(self.client_sizes[cid])

            # Aggregate
            new_state = fedavg(
                self.global_hsbn.state_dict(),
                client_states,
                client_weights,
            )
            self.global_hsbn.load_state_dict(new_state)

            # Evaluate
            acc = self.evaluate()
            log = {
                "round"       : round_idx,
                "top1_acc"    : acc,
                "client_logs" : round_metrics,
            }
            self.history.append(log)
            print(f"Round {round_idx:03d} | Top-1 Acc: {acc:.4f} | "
                  f"Clients: {list(selected)}")

        return self.history

    @torch.no_grad()
    def evaluate(self) -> float:
        """Global test accuracy using HSBN coarse logits as proxy."""
        self.global_hsbn.eval()
        correct = total = 0

        for x, fine_labels in self.test_loader:
            x           = x.to(self.device)
            fine_labels = fine_labels.to(self.device)

            from fl.client.base_client import fine_to_coarse
            coarse_labels = fine_to_coarse(fine_labels)

            out = self.global_hsbn(x, fine_labels, coarse_labels)
            preds = out.fine_logits.argmax(dim=1)
            correct += (preds == fine_labels).sum().item()
            total   += fine_labels.size(0)

        self.global_hsbn.train()
        return correct / total if total > 0 else 0.0