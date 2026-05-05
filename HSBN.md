# HSBN (Hierarchical Stochastic Bottleneck Network)

## Overview

HSBN is a hierarchical neural architecture built around stochastic bottleneck channels and top-down feedback. It replaces the previous FIN naming and is designed to learn multi-level abstractions through a structured bottom-up / top-down forward pass.

## Key concepts

- **Levels**: The model is organized into Level 0, Level 1, and Level 2.
- **Bottleneck channels**: Stochastic compression channels enforce bandwidth constraints between levels.
- **Top-down refinement**: Higher levels send feedback to refine lower-level representations before objectives are computed.
- **Hierarchical loss**: HSBN combines per-level objectives with bandwidth penalty terms.

## Structure

- `hsbn/network/hsbn.py`
  - Main model assembly and forward pass implementation.
  - Defines `HSBN`, `HSBNOutput`, and `build_hsbn()`.
- `hsbn/levels/level0.py`
  - Raw signal processing level with encoder, decoder, and feedback gate.
- `hsbn/levels/level1.py`
  - Mid-level transformer-like processing with fine classification.
- `hsbn/levels/level2.py`
  - Apex-level MLP and coarse classification.
- `hsbn/channels/bottleneck.py`
  - Stochastic bandwidth-limited channel implementation.
- `hsbn/losses/hierarchical.py`
  - Hierarchical loss assembly and breakdown tracking.
- `hsbn/utils/annealing.py`
  - Beta annealing scheduler for channel bandwidth budgets.

## Usage

Instantiate the model via the factory:

```python
from hsbn.network.hsbn import build_hsbn

cfg = ...  # parsed config dict
model = build_hsbn(cfg)
```

Then train with the returned model and use `model.update_betas(epoch)` to anneal channel bandwidth during training.

## Notes

- The package is renamed from `fin` to `hsbn` throughout the source tree.
- Internal references and type names are updated to reflect `HSBN` and `L_HSBN`.
