
import torch
import torch.nn as nn

from feed_forward import FeedForward


from multi_head_attention import MultiHeadAttention
from mask import create_causal_mask


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model, num_heads
        )

        self.cross_attention = MultiHeadAttention(
            d_model, num_heads
        )

        self.feed_forward = FeedForward(
            d_model, d_ff
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        # Always enforce causality, including when a padding mask is supplied.
        mask = create_causal_mask(x.size(1), x.device)
        if tgt_mask is not None:
            mask = mask & tgt_mask

        attn, _ = self.self_attention(x, mask=mask)
        x = self.norm1(x + attn)

        # 2. Cross-Attention
        attn, _ = self.cross_attention(
            x, encoder_output, encoder_output, mask=src_mask
        )
        x = self.norm2(x + attn)

        # 3. Feed-Forward
        ff = self.feed_forward(x)
        x = self.norm3(x + ff)

        return x


if __name__ == "__main__":
    decoder = TransformerDecoderLayer(
        d_model=8,
        num_heads=2,
        d_ff=32
    )

    x = torch.randn(1, 3, 8)
    encoder_output = torch.randn(1, 3, 8)

    output = decoder(x, encoder_output)

    print("Decoder input shape:", x.shape)
    print("Encoder output shape:", encoder_output.shape)
    print("Decoder output shape:", output.shape)
    print("Transformer Decoder successfully built!")
