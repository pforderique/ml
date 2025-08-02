"""This module provides a BPE tokenizer using the Tokenizers library."""

from tokenizers import Tokenizer

class BPETokenizer:
    """A class for handling Byte-Pair Encoding (BPE) tokenization."""

    def __init__(self, tokenizer_path="layers/tokenizer/tokenizer.json"):
        self.tokenizer: Tokenizer = Tokenizer.from_file(tokenizer_path)

    def encode(self, text: str) -> list[int]:
        """Encode a text string into token IDs."""
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs into a text string."""
        return self.tokenizer.decode(ids)

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size of the tokenizer."""
        return self.tokenizer.get_vocab_size()
