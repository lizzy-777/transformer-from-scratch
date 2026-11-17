from pathlib import Path

import torch

from transformer import Transformer
from train import EOS, collate, greedy_decode


def main():
    torch.set_num_threads(1)

    checkpoint_path = (
        Path(__file__).resolve().parent
        / "checkpoints"
        / "step9_copy_task.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model = Transformer(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Model loaded successfully.")

    examples = [
        [4, 5, 6],
        [13, 14, 9, 13],
        [12, 14, 3, 6, 3, 12],
    ]

    with torch.inference_mode():
        for row in examples:
            src, _, _ = collate([row], "cpu")
            prediction = greedy_decode(
                model, src, max_steps=len(row) + 1
            )[0].tolist()

            expected = [*row, EOS]
            print(f"Input:     {row}")
            print(f"Expected:  {expected}")
            print(f"Predicted: {prediction}")
            print(f"Correct:   {prediction == expected}")
            print()


if __name__ == "__main__":
    main()