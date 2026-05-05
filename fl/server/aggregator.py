# fl/server/aggregator.py

import copy
import torch
from typing import List, Dict


def fedavg(
    global_state : dict,
    client_states: List[dict],
    client_weights: List[float] | None = None,
) -> dict:
    """
    Weighted FedAvg over HSBN state dicts.

    Args:
        global_state   : current global HSBN state_dict (used as base)
        client_states  : list of state_dicts from participating clients
        client_weights : per-client weights (e.g. dataset size).
                         If None, uniform averaging.

    Returns:
        new aggregated state_dict
    """
    if not client_states:
        return global_state

    n = len(client_states)
    if client_weights is None:
        client_weights = [1.0 / n] * n
    else:
        total = sum(client_weights)
        client_weights = [w / total for w in client_weights]

    new_state = copy.deepcopy(client_states[0])

    for key in new_state:
        new_state[key] = torch.zeros_like(new_state[key], dtype=torch.float32)
        for state, weight in zip(client_states, client_weights):
            new_state[key] += weight * state[key].float()

    return new_state