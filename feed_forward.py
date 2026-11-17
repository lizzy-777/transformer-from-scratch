
import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    x = torch.randn(1, 3, 8)

    model = FeedForward(
        d_model=8,
        d_ff=32
    )

    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Feed Forward Output:")
    print(output)
