"""Token and positional embeddings for transformer models."""

import torch
from torch import nn


class TokenAndPositionalEmbedding(nn.Module):
    """Token and positional embeddings for transformer models."""

    def __init__(self, vocab_size, d_model, max_seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)  # [V, D]
        self.pos_emb = nn.Embedding(max_seq_len, d_model)  # [T_max, D]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (batch_size, seq_len) [B, T]
        returns: (batch_size, seq_len, d_model) [B, T, D]
        """
        seq_len = input_ids.size(1)
        positions = torch.arange(
            seq_len,
            device=input_ids.device
        ).unsqueeze(0)  # [1, seq_len]

        token_embeddings = self.token_emb(input_ids)     # [B, T, D]
        position_embeddings = self.pos_emb(positions)    # [1, T, D]

        return token_embeddings + position_embeddings  # Broadcasted [B, T, D]
