
import torch


def create_causal_mask(seq_len, device="cpu"):
    """
    Create a causal attention mask.

    True  = allowed to attend
    False = blocked
    """
    mask = torch.tril(
        torch.ones(
            seq_len,
            seq_len,
            device=device,
            dtype=torch.bool
        )
    )

    return mask


if __name__ == "__main__":
    seq_len = 4

    mask = create_causal_mask(seq_len)

    print("Causal Mask:")
    print(mask)

    print("Mask shape:", mask.shape)
