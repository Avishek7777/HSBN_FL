# fl/client/base_client.py

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from hsbn.network.hsbn import HSBN


class FLClient:
    """
    A single federated learning client.

    Holds:
        - A local copy of the shared HSBN (the federated backbone)
        - A private LocalHead that receives z2 from HSBN and produces logits
        - Its own DataLoader (non-IID partition)

    Training strategy:
        1. Run HSBN forward → get HSBNOutput (total_loss + z2)
        2. Pass z2 to local head → get local logits → local head loss
        3. Backprop through combined loss
        4. Only HSBN weights are sent back to the server
    """

    def __init__(
        self,
        client_id   : int,
        hsbn        : HSBN,
        local_head  : nn.Module,
        dataloader  : DataLoader,
        arch_name   : str,
        num_classes : int,
        lr          : float = 1e-3,
        local_epochs: int   = 5,
        device      : str   = "cpu",
        head_loss_weight: float = 0.5,
    ):
        self.client_id   = client_id
        self.arch_name   = arch_name
        self.device      = device
        self.local_epochs = local_epochs
        self.head_loss_weight = head_loss_weight

        # HSBN is a fresh copy per client (loaded from server state_dict each round)
        self.hsbn = hsbn.to(device)
        self.local_head = local_head.to(device)

        self.dataloader = dataloader
        self.criterion  = nn.CrossEntropyLoss()

        # Optimizer covers both HSBN and local head
        self.optimizer = optim.Adam(
            list(self.hsbn.parameters()) + list(self.local_head.parameters()),
            lr=lr,
        )

    def load_hsbn_state(self, state_dict: dict):
        """Receive global HSBN weights from server before local training."""
        self.hsbn.load_state_dict(state_dict)

    def train_round(self, global_epoch: int) -> dict:
        """
        Run local_epochs of training for this round.

        Args:
            global_epoch: current global round (for beta annealing)

        Returns:
            metrics dict with avg losses
        """
        self.hsbn.train()
        self.local_head.train()
        self.hsbn.update_betas(global_epoch)

        total_hsbn_loss = 0.0
        total_head_loss = 0.0
        num_batches = 0

        for _ in range(self.local_epochs):
            for batch in self.dataloader:
                x, (fine_labels, coarse_labels) = self._unpack_batch(batch)
                x            = x.to(self.device)
                fine_labels  = fine_labels.to(self.device)
                coarse_labels = coarse_labels.to(self.device)

                self.optimizer.zero_grad()

                # HSBN forward — uses fine + coarse labels for its internal losses
                hsbn_out = self.hsbn(x, fine_labels, coarse_labels)

                # Local head forward on z2 (detached — head doesn't backprop into HSBN)
                # Use fine_labels as the local task target
                local_logits = self.local_head(hsbn_out.z2.detach())
                head_loss = self.criterion(local_logits, fine_labels)

                # Combined loss: HSBN hierarchical loss + weighted head loss
                loss = hsbn_out.total_loss + self.head_loss_weight * head_loss
                loss.backward()
                self.optimizer.step()

                total_hsbn_loss += hsbn_out.total_loss.item()
                total_head_loss += head_loss.item()
                num_batches += 1

        return {
            "client_id"      : self.client_id,
            "arch"           : self.arch_name,
            "avg_hsbn_loss"  : total_hsbn_loss / max(num_batches, 1),
            "avg_head_loss"  : total_head_loss / max(num_batches, 1),
            "num_batches"    : num_batches,
        }

    def get_hsbn_state(self) -> dict:
        """Return HSBN weights for aggregation. Local head stays private."""
        return copy.deepcopy(self.hsbn.state_dict())

    def _unpack_batch(self, batch):
        """
        CIFAR-100 via torchvision returns (x, fine_label).
        Coarse labels must be derived or loaded separately.
        This method centralizes that unpacking logic.
        """
        x, fine_labels = batch
        # CIFAR-100 coarse labels: derive from fine via standard mapping
        coarse_labels = fine_to_coarse(fine_labels)
        return x, (fine_labels, coarse_labels)


# CIFAR-100 fine → coarse label mapping (20 superclasses)
_FINE_TO_COARSE = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11,
    5, 5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4,
    18, 17, 10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9,
    13, 15, 13, 16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17,
    8, 14, 13,
]
_COARSE_TENSOR = torch.tensor(_FINE_TO_COARSE, dtype=torch.long)

def fine_to_coarse(fine_labels: torch.Tensor) -> torch.Tensor:
    return _COARSE_TENSOR[fine_labels.cpu()].to(fine_labels.device)
