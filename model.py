
import torch
import torch.nn.functional as F
import math

def self_attention(Q, K, V):
    # Dimension of Key vectors
    d_k = Q.size(-1)

    # Step 1: Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1))
    scores = scores / math.sqrt(d_k)

    # Step 2: Apply softmax
    weights = F.softmax(scores, dim=-1)

    # Step 3: Compute weighted output
    output = torch.matmul(weights, V)

    return output, weights


# Example: 3 tokens, each with 4 features
Q = torch.randn(1, 3, 4)
K = torch.randn(1, 3, 4)
V = torch.randn(1, 3, 4)

output, weights = self_attention(Q, K, V)

print("Attention Output:")
print(output)

print("\nAttention Weights:")
print(weights)
