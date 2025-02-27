import os
from functools import cached_property
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class TrainingDataLoader:
    """
    Dataloader based on the one from nanoGPT. Each batch successively walks the shards in a dataset.
    """

    def __init__(
        self,
        dir_path: str,
        B: int,
        T: int,
        process_rank: int,
        num_processes: int,
        split: str,
        use_random_offsets: bool = False,
    ):
        """
        dir_path: Path to the directory containing the dataset shards.
        B: Batch size.
        T: Sequence length.
        process_rank: Rank of the current process.
        num_processes: Total number of processes.
        split: Dataset split to use (e.g. "train", "val").
        use_random_offsets: If True, samples will be generated using random offsets.
        """
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}
        self.split = split

        # Should samples be generated using random offsets?
        self.use_random_offsets = use_random_offsets

        # Get the shard filenames
        shards = os.listdir(dir_path)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(dir_path, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"No shards found for split {split}"
        print(f"Found {len(shards)} shards for split {split}")

        # Should be fast - memory mapped and shape is in the header
        self.total_tokens = sum([np.load(s, mmap_mode="r").size for s in self.shards])

        self.reset()

    def reset(self):
        """
        Reset the dataloader to the beginning of the dataset.
        """
        # Use a fixed seed for deterministic offsets if using random offsets
        self.generator = torch.Generator()
        self.generator.manual_seed(42)

        # Current shard will be set to 0 after loading the "next" shard
        self.current_shard_idx = -1
        self.load_next_shard()

    def load_next_shard(self):
        """
        Load the next shard in the dataset.
        """
        # Update the current shard index
        self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shards)

        # Choose a new offset if using random offsets
        if self.use_random_offsets:
            self.offset = torch.randint(0, self.T, (1,), generator=self.generator).item()
        else:
            self.offset = 0

        # Load shard data
        self.tokens = self.load_tokens(self.shards[self.current_shard_idx])
        self.current_position = self.B * self.T * self.process_rank + self.offset

    def next_batch(self, device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a batch of size B x T.
        """
        B, T = self.B, self.T
        start_idx = self.current_position
        end_idx = self.current_position + B * T + 1
        buf = self.tokens[start_idx:end_idx]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.load_next_shard()
        x, y = x.to(device), y.to(device)
        return x, y

    def load_tokens(self, filename) -> torch.Tensor:
        """
        Loads a numpy array of tokens from a file.
        """
        npt = np.load(filename, allow_pickle=False)
        npt = npt.astype(np.int32)
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt


class DatasetShard:
    """
    Tokens from a specific dataset shard.
    """

    def __init__(
        self,
        dir_path: Path,
        split: str,
        shard_idx: int = 0,
        limit: Optional[int] = None,
    ):
        """
        dir_path: Path to the directory containing the dataset shards.
        split: Dataset split to use (e.g. "train", "val").
        shard_idx: Index of the shard to load.
        """
        # Get the shard filenames
        shard_paths = os.listdir(dir_path)
        assert split in {"train", "val"}
        shard_paths = [s for s in shard_paths if split in s]
        shard_paths = sorted(shard_paths)
        shard_paths = [os.path.join(dir_path, s) for s in shard_paths]
        assert len(shard_paths) > 0, f"No shards found for split {split}"
        assert 0 <= shard_idx < len(shard_paths), f"Shard index {shard_idx} out of bounds"
        self.split = split
        self.shard_idx = shard_idx
        self.shard_path = shard_paths[shard_idx]
        self.limit = limit

    @cached_property
    def tokens(self) -> torch.Tensor:
        """
        Defer loading tokens until first access.
        """
        npt = np.load(self.shard_path, allow_pickle=False)
        npt = npt.astype(np.int32)
        if self.limit is not None:
            npt = npt[: self.limit]
        return torch.tensor(npt, dtype=torch.long)
