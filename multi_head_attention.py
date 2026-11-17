
import math
import torch
import torch.nn as nn

from mask import create_causal_mask


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.shape

        return x.reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.d_k
        ).transpose(1, 2)

    def forward(self, query, key=None, value=None, mask=None):

        if key is None:
            key = query

        if value is None:
            value = key

        batch_size, query_len, _ = query.shape

        # 1. Linear projections
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        # 2. Scaled dot-product attention
        scores = torch.matmul(
            Q, K.transpose(-2, -1)
        ) / math.sqrt(self.d_k)

        # 3. Attention mask
        if mask is not None:
            mask = mask.to(
                device=scores.device,
                dtype=torch.bool
            )

            scores = scores.masked_fill(
                ~mask,
                float("-inf")
            )

            if (~mask).all(dim=-1).any():
                raise ValueError(
                    "Every query must have at least one "
                    "unmasked key."
                )

        # 4. Softmax
        weights = torch.softmax(scores, dim=-1)

        # 5. Weighted sum
        attention = torch.matmul(weights, V)

        # 6. Combine heads
        attention = attention.transpose(1, 2)

        attention = attention.contiguous().reshape(
            batch_size,
            query_len,
            self.d_model
        )

        # 7. Output projection
        output = self.W_o(attention)

        return output, weights


if __name__ == "__main__":

    torch.manual_seed(42)

    model = MultiHeadAttention(
        d_model=8,
        num_heads=2
    )

    x = torch.randn(1, 3, 8)

    mask = create_causal_mask(
        seq_len=x.size(1),
        device=x.device
    )

    # Test 1: Masked Self-Attention
    output, weights = model(x, mask=mask)

    print("Self-Attention output:", output.shape)
    print("Attention weights:", weights.shape)

    assert torch.all(
        weights.masked_select(~mask) == 0
    )

    print("Causal Mask successfully applied!")

    # Test 2: Cross-Attention
    encoder_output = torch.randn(1, 5, 8)

    cross_output, cross_weights = model(
        x,
        encoder_output,
        encoder_output
    )

    print("Cross-Attention output:", cross_output.shape)
    print("Cross-Attention weights:", cross_weights.shape)

    assert cross_output.shape == (1, 3, 8)
    assert cross_weights.shape == (1, 2, 3, 5)

    print("Multi-Head Attention successfully built!")
