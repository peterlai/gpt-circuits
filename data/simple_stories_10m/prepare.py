"""
Downloads, tokenizes, and saves SimpleStories dataset to disk (limited to about 10M tokens).
https://huggingface.co/datasets/lennart-finke/SimpleStories

$ python -m data.simple_stories_10m.prepare
"""

import os

from datasets import Dataset, load_dataset

from data.tokenizers import TikTokenTokenizer
from data.utils import save_dataset

if __name__ == "__main__":
    out_dir = os.path.dirname(__file__)
    train_dataset: Dataset = load_dataset("lennart-finke/SimpleStories", split="train")  # type: ignore
    val_dataset: Dataset = load_dataset("lennart-finke/SimpleStories", split="test")  # type: ignore

    # Limit the number of examples
    train_dataset = train_dataset.select(range(35000))
    val_dataset = val_dataset.select(range(3500))

    # Use about half the available CPUs for tokenization
    num_proc = max(1, (os.cpu_count() or 2) // 2)

    # Use the TikToken tokenizer
    tokenizer = TikTokenTokenizer()

    # Tokenization function
    def tokenize(example):
        # Add an end-of-text token after every story
        ids = tokenizer.encode(example["story"])
        ids.append(tokenizer.encode("<|endoftext|>")[0])
        return {"ids": ids}

    # Tokenize and save the training dataset (~10M tokens)
    train_dataset = train_dataset.map(tokenize, num_proc=num_proc)
    save_dataset(train_dataset, out_dir, "train", num_shards=1)

    # Tokenize and save the validation dataset (~1M tokens)
    val_dataset = val_dataset.map(tokenize, num_proc=num_proc)
    save_dataset(val_dataset, out_dir, "val", num_shards=1)
