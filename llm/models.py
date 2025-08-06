"""Transformer-based Language Models"""

import torch
from torch import nn

from layers.transformer_block import TransformerBlock
from layers.embedding import TokenAndPositionalEmbedding


class MiniLLM(nn.Module):
    """Minimal Transformer-based Language Model."""

    def __init__(self, vocab_size, d_model=256, n_layers=4, n_heads=2, max_seq_len=128):
        super().__init__()

        self.embed = TokenAndPositionalEmbedding(
            vocab_size, d_model, max_seq_len
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, max_seq_len)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""

        x = self.embed(input_ids)  # [B, T, D]

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # [B, T, vocab_size]
        return logits
