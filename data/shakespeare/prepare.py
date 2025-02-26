"""
Processes Shakespeare dataset a using a character-based tokenizer.
https://huggingface.co/datasets/karpathy/tiny_shakespeare

$ python -m data.shakespeare.prepare
"""

import os

from datasets import Dataset

from data.tokenizers import ASCIITokenizer
from data.utils import save_dataset

if __name__ == "__main__":
    out_dir = os.path.dirname(__file__)

    # Load corpus from text file
    shakespeare_corpus = open(f"{out_dir}/input.txt", "r").read()

    # Split the corpus into training and validation data
    train_length = int(len(shakespeare_corpus) * 0.9)
    trimmed_train_length = train_length - (train_length % 256)  # Round down to nearest multiple of 256
    train_dataset = Dataset.from_list([{"text": shakespeare_corpus[:trimmed_train_length]}])  # 1,003,776 tokens
    val_dataset = Dataset.from_list([{"text": shakespeare_corpus[trimmed_train_length:]}])  # 111,618 tokens

    # Use about half the available CPUs for tokenization
    num_proc = max(1, (os.cpu_count() or 2) // 2)

    # Use the ASCII tokenizer
    tokenizer = ASCIITokenizer()

    # Tokenization function
    def tokenize(example):
        ids = tokenizer.encode(example["text"])
        return {"ids": ids}

    # Tokenize and save the training dataset
    train_dataset = train_dataset.map(tokenize, num_proc=num_proc)
    save_dataset(train_dataset, out_dir, "train", num_shards=1)

    # Tokenize and save the validation dataset
    val_dataset = val_dataset.map(tokenize, num_proc=num_proc)
    save_dataset(val_dataset, out_dir, "val", num_shards=1)
