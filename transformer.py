
import torch
import torch.nn as nn

from encoder import TransformerEncoderLayer
from decoder import TransformerDecoderLayer
from positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=128,
        num_heads=4,
        d_ff=512,
        num_layers=2,
        max_len=5000,
        padding_idx=None
    ):
        super().__init__()

        self.d_model = d_model
        self.padding_idx = padding_idx

        # 1. Token Embedding
        self.src_embedding = nn.Embedding(
            src_vocab_size, d_model, padding_idx=padding_idx
        )
        self.tgt_embedding = nn.Embedding(
            tgt_vocab_size, d_model, padding_idx=padding_idx
        )

        # 2. Positional Encoding
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_len
        )

        # 3. Encoder Layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff
            )
            for _ in range(num_layers)
        ])

        # 4. Decoder Layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff
            )
            for _ in range(num_layers)
        ])

        # 5. Output Projection
        self.fc_out = nn.Linear(
            d_model, tgt_vocab_size
        )

    def forward(self, src, tgt):
        # Mask keys only: padded queries are ignored by the training loss.
        src_mask = tgt_mask = None
        if self.padding_idx is not None:
            src_mask = (src != self.padding_idx)[:, None, None, :]
            tgt_mask = (tgt != self.padding_idx)[:, None, None, :]


        # Embedding + Positional Encoding
        src = self.positional_encoding(
            self.src_embedding(src)
        )

        tgt = self.positional_encoding(
            self.tgt_embedding(tgt)
        )

        # Encoder
        encoder_output = src

        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, mask=src_mask)

        # Decoder
        decoder_output = tgt

        for layer in self.decoder_layers:
            decoder_output = layer(
                decoder_output,
                encoder_output,
                tgt_mask=tgt_mask,
                src_mask=src_mask
            )

        # Vocabulary Prediction
        logits = self.fc_out(decoder_output)

        return logits


if __name__ == "__main__":

    model = Transformer(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        d_model=8,
        num_heads=2,
        d_ff=32,
        num_layers=2
    )

    # Example token IDs
    src = torch.randint(0, 1000, (1, 5))
    tgt = torch.randint(0, 1000, (1, 4))

    output = model(src, tgt)

    print("Source shape:", src.shape)
    print("Target shape:", tgt.shape)
    print("Transformer output shape:", output.shape)

    print("Transformer successfully built!")
