"""
This module contains classes for tokenizing text.
"""

from enum import Enum
from typing import Protocol, Literal

import tiktoken


class TokenizerType(str, Enum):
    """
    Type of tokenizer.
    """

    TIKTOKEN = "tiktoken"
    ASCII = "ascii"

    def as_tokenizer(self) -> "Tokenizer":
        """
        Returns the tokenizer corresponding to this type.
        """
        if self == TokenizerType.TIKTOKEN:
            return TikTokenTokenizer()
        elif self == TokenizerType.ASCII:
            return ASCIITokenizer()
        else:
            raise ValueError(f"Invalid tokenizer type: {self}")


class Tokenizer(Protocol):
    """
    Tokenizer interface.
    """

    # The size of the vocabulary.
    vocab_size: int
    padding_side : Literal["left", "right"] = "right"

    def encode(self, text) -> list[int]:
        """
        Tokenizes the input text and returns a list of token IDs.
        """
        ...

    def decode_sequence(self, tokens: list[int]) -> str:
        """
        Converts a list of token IDs to a string.
        """
        ...

    def decode_token(self, token: int) -> str:
        """
        Converts a single token ID to a string.
        """
        ...


class TikTokenTokenizer(Tokenizer):
    """
    Tokenizer for TikToken.
    """

    # 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    vocab_size = 50257

    def __init__(self):
        self.encoding = tiktoken.get_encoding("gpt2")

    def encode(self, text) -> list[int]:
        return self.encoding.encode(text, allowed_special="all")

    def decode_sequence(self, tokens: list[int]) -> str:
        # Note that multiple tokens may have been used to represent a single UTF-8 character.
        return self.encoding.decode(tokens)

    def decode_token(self, token: int) -> str:
        # Decoding falls back to a hex representation if the result isn't printable.
        s = self.encoding.decode_single_token_bytes(token)
        try:
            return s.decode("utf-8")
        except Exception as _:
            return "".join([f"\\x{d:02x}" for d in s])


class ASCIITokenizer(Tokenizer):
    """
    Tokenizer that treats each character in the input as a token.
    """

    vocab_size = 128

    def encode(self, text) -> list[int]:
        # Note that invalid characters are replaced with "?".
        return [ord(c) if ord(c) < 128 else ord("?") for c in text]

    def decode_sequence(self, tokens: list[int]) -> str:
        return "".join([chr(b) for b in tokens])

    def decode_token(self, token: int) -> str:
        return chr(token)
