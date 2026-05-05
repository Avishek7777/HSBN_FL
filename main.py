# main.py

import argparse
import yaml
import torch
from fl.runner import FLRunner


def main():
    parser = argparse.ArgumentParser(description="HSBN Federated Learning")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment config YAML")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    runner = FLRunner(cfg, device=args.device)
    runner.run()


if __name__ == "__main__":
    main()