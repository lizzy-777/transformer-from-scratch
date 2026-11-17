
import torch
import torch.nn as nn

from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads
        )

        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=d_ff
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Step 1: Multi-Head Attention
        attention_output, weights = self.attention(x, mask=mask)

        # Step 2: Residual Connection + LayerNorm
        x = self.norm1(x + attention_output)

        # Step 3: Feed-Forward Network
        ff_output = self.feed_forward(x)

        # Step 4: Residual Connection + LayerNorm
        x = self.norm2(x + ff_output)

        return x


if __name__ == "__main__":
    x = torch.randn(1, 3, 8)

    encoder = TransformerEncoderLayer(
        d_model=8,
        num_heads=2,
        d_ff=32
    )

    output = encoder(x)

    print("Input shape:", x.shape)
    print("Encoder output shape:", output.shape)
    print("Transformer Encoder successfully built!")
