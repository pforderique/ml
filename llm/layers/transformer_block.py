"""Causal Self-Attention and Transformer Block Implementation"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class CausalSelfAttention(nn.Module):
    """Causal Self-Attention Layer for Transformer Models."""

    def __init__(self, d_model, n_heads, max_seq_len):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.n_heads = n_heads

        # Vaswani et al. (2017) sets d_k = d_v = d_model // n_heads
        # However, in practice, d_k = d_v = d_model is often used - allowing us
        # to use a single linear layer for Q, K, V projections.
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)

        # W^O projection
        # In the paper it is listed as [h*d_v, d_model]
        # Since d_v = d_model / n_heads, we have W^O = [d_model, d_model]
        self.out_proj = nn.Linear(d_model, d_model)

        # Causal mask: [1, 1, T_max, T_max]
        self.mask: torch.Tensor
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len))
            .unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model) [B, T, D]
        returns: (batch_size, seq_len, d_model) [B, T, D]
        """
        # pylint: disable=invalid-name
        B, T, D = x.shape

        # [B, T, D] → x@(W^Q), x@(W^K), x@(W^V) → Q, K, V projections [B, T, D]
        qkv: torch.Tensor = self.qkv_proj(x)  # [B, T, 3D]

        # Split into Q, K, V: [B, T, 3D] → [B, T, D], [B, T, D], [B, T, D]
        Q, K, V = qkv.chunk(3, dim=-1)

        # Split heads: [B, T, h, D//h] → [B, h, T, D//h], e.g. W^Q of ith head
        h = self.n_heads
        d_h = D // h
        Q = Q.view(B, T, h, d_h).transpose(1, 2)
        K = K.view(B, T, h, d_h).transpose(1, 2)
        V = V.view(B, T, h, d_h).transpose(1, 2)

        # Scaled Dot-Product Attention
        # Pytorch performs batched multiplication with [B, h, T, D//h] @ [B, h, D//h, T]
        attention = Q @ K.transpose(-2, -1) / math.sqrt(d_h)  # [B, h, T, T]

        # Apply causal mask: [B, h, T, T] → [B, h, T, T]
        # Masking future positions in the attention matrix with -inf since
        # softmax(-inf) = 0, ensuring no information leaks from future tokens.
        # Only use current T (<= T_max) of the lower triangular mask
        attention = attention.masked_fill(
            self.mask[:, :, :T, :T] == 0, float("-inf")
        )

        attn_weights = F.softmax(attention, dim=-1)  # still [B, h, T, T]

        # [B, h, T, T] @ [B, h, T, D//h] → [B, h, T, D//h]
        attn_output = attn_weights @ V

        # Concatenate heads: [B, h, T, D//h] → [B, T, D]
        # Note: Transpose does not change the memory layout (it only changes the
        # strides, i.e. how to step through the tensor). Therefore, we use
        # contiguous() to ensure the output is contiguous in memory before
        # reshaping using view.
        attn_output = (attn_output
                       .transpose(1, 2)
                       .contiguous()
                       .view(B, T, D))  # [B, T, D]

        # Final linear projection: [B, T, D] → [B, T, D]
        # This mixes and transforms the outputs of all heads.
        return self.out_proj(attn_output)


class TransformerBlock(nn.Module):
    """Transformer Block with Self-Attention and Feed-Forward Network."""

    def __init__(self, d_model, n_heads, ff_hidden_dim, max_seq_len):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.GELU(),
            nn.Linear(ff_hidden_dim, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass through the transformer block."""

        # Original paper uses Post-Norm, but in practice (GPT, LLaMa) Pre-Norm
        # is often preferred.
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
