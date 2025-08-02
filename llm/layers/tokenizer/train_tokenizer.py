"""Train a Byte-Pair Encoding (BPE) tokenizer using the Tokenizers library."""

import argparse
from collections.abc import Iterable
import os
import pathlib

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors


Path = pathlib.Path
_parent_dir = Path(__file__).parent.resolve()


def read_all_txt_files(corpus_dir: Path) -> Iterable[str]:
    """Read all .txt files in the given directory and yield their contents."""
    for fname in os.listdir(corpus_dir):
        if fname.endswith(".txt"):
            with open(os.path.join(corpus_dir, fname), 'r', encoding='utf-8') as f:
                yield f.read()


def train_bpe_tokenizer(corpus_path: Path, output: Path, vocab_size: int):
    """Train a BPE tokenizer on the given corpus."""

    tokenizer = Tokenizer(models.BPE())
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,  # type: ignore
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]  # type: ignore
    )

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()  # type: ignore
    tokenizer.decoder = decoders.ByteLevel()  # type: ignore
    tokenizer.post_processor = processors.ByteLevel(
        trim_offsets=True)  # type: ignore

    corpus_iter = read_all_txt_files(corpus_path)
    tokenizer.train_from_iterator(corpus_iter, trainer=trainer)
    tokenizer.save(output)
    print(f"✅ Saved tokenizer to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer.")
    parser.add_argument(
        "--corpus",
        type=str,
        default=(_parent_dir / "corpus").as_posix(),
        help="Path to the corpus text file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=(_parent_dir / "tokenizer.json").as_posix(),
        help="Path to save the trained tokenizer."
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=1000,
        help="Vocabulary size for the tokenizer."
    )
    args = parser.parse_args()

    train_bpe_tokenizer(args.corpus, args.output, args.vocab_size)
    print("BPE tokenizer training complete.")
