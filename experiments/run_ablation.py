# experiments/run_ablation.py

import yaml
import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from fl.runner import FLRunner


CONFIG_FILES = [
    "configs/dirichlet_01.yaml",
    "configs/dirichlet_05.yaml",
    "configs/dirichlet_10.yaml",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_all():
    os.makedirs("results", exist_ok=True)

    for config_path in CONFIG_FILES:
        cfg = load_config(config_path)
        exp_name = cfg.get("experiment", {}).get("name", os.path.basename(config_path))
        print(f"\n{'='*60}")
        print(f"Running experiment: {exp_name}")
        print(f"  alpha = {cfg['data']['dirichlet_alpha']}")
        print(f"{'='*60}")

        runner = FLRunner(cfg, device=DEVICE)
        history = runner.run()

        out_path = f"results/{exp_name}_history.json"
        with open(out_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    run_all()