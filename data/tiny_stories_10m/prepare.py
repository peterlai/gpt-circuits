"""
Produces a dataset with approximately 10 million tokens based on the TinyStories dataset.
https://huggingface.co/datasets/roneneldan/TinyStories

$ python -m data.tiny_stories_10m.prepare
"""

import os

from datasets import Dataset, load_dataset

from data.tokenizers import TikTokenTokenizer
from data.utils import save_dataset

if __name__ == "__main__":
    out_dir = os.path.dirname(__file__)
    data_files = {
        # Use updated GPT4 version of the dataset
        "train": "TinyStoriesV2-GPT4-train.txt",
        "validation": "TinyStoriesV2-GPT4-valid.txt",
    }
    train_dataset: Dataset = load_dataset("roneneldan/TinyStories", data_files=data_files, split="train")  # type: ignore
    val_dataset: Dataset = load_dataset("roneneldan/TinyStories", data_files=data_files, split="validation")  # type: ignore

    # Limit the number of examples
    train_dataset = train_dataset.select(range(286000))
    val_dataset = val_dataset.select(range(28600))

    # Use about half the available CPUs for tokenization
    num_proc = max(1, (os.cpu_count() or 2) // 2)

    # Use the TikToken tokenizer
    tokenizer = TikTokenTokenizer()

    # Tokenization function
    def tokenize(example):
        # Add a new line token after every paragraph. We don't need to add an end-of-text token
        # because those already exist in the dataset.
        ids = tokenizer.encode(example["text"])
        ids.append(tokenizer.encode("\n")[0])
        return {"ids": ids}

    # Tokenize and save the training dataset
    train_dataset = train_dataset.map(tokenize, num_proc=num_proc)
    save_dataset(train_dataset, out_dir, "train", num_shards=1)

    # Tokenize and save the validation dataset
    val_dataset = val_dataset.map(tokenize, num_proc=num_proc)
    save_dataset(val_dataset, out_dir, "val", num_shards=1)
