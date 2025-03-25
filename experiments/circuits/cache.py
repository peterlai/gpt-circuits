"""
Cache feature metrics.

$ python -m experiments.circuits.cache
$ python -m experiments.circuits.cache --model=e2e.jumprelu-staircase.shakespeare_64x4
"""

import argparse
from pathlib import Path

from circuits.features import compute_metrics
from config import TrainingConfig
from data.dataloaders import DatasetShard


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="e2e.jumprelu.shakespeare_64x4", help="Model to analyze")
    parser.add_argument("--data_dir", type=str, default="data/shakespeare", help="Dataset split to use")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--shard_idx", type=int, default=0, help="Shard to load data from")
    parser.add_argument("--limit", type=int, default=1e7, help="Max number of tokens to analyze")
    parser.add_argument("--batch_size", type=int, default=256, help="Max number of tokens to analyze")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Load tokens
    shard = DatasetShard(
        dir_path=Path(args.data_dir),
        split=args.split,
        shard_idx=args.shard_idx,
        limit=int(args.limit),
    )

    # Cache feature metrics
    compute_metrics(TrainingConfig.checkpoints_dir / args.model, shard, args.batch_size)
