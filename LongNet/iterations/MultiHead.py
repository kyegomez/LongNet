import torch
from torch import nn
import torch.nn.functional as F

class MultiHeadDilatedAttention(nn.Module):
    def __init__(self, d_model, num_heads, segment_size, dilation_rate, parallel=False, distributed=False):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.segment_size = segment_size
        self.dilation_rate = dilation_rate
        self.parallel = parallel
        self.distributed = distributed

        self.head_dim = d_model // num_heads

        assert (
            self.head_dim * num_heads == d_model
        ), "Embedding dimension should be divisible by number of heads"

        self.scaling = self.head_dim ** -0.5

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        # Calculate the necessary padding
        padding_len = -seq_len % self.segment_size
        Q = F.pad(Q, (0, 0, 0, padding_len))
        K = F.pad(K, (0, 0, 0, padding_len))
        V = F.pad(V, (0, 0, 0, padding_len))

        # Calculate num of segments
        num_segments = Q.size(1) // self.segment_size

        Q = Q.view(batch_size, num_segments, self.segment_size, self.num_heads, self.head_dim)
        K = K.view(batch_size, num_segments, self.segment_size, self.num_heads, self.head_dim)
        V = V.view(batch_size, num_segments, self.segment_size, self.num_heads, self.head_dim)

        # Initialize output tensor
        outputs = torch.zeros_like(x)

        # Perform attention for each head
        for j in range(self.num_heads):
            sj = j % self.dilation_rate
            Q_sj = Q[:, :, sj::self.dilation_rate, j]
            K_sj = K[:, :, sj::self.dilation_rate, j]
            V_sj = V[:, :, sj::self.dilation_rate, j]

            # Compute attention scores
            attn_scores = torch.einsum("bnsl,bnsl->bns", Q_sj, K_sj) * self.scaling

            # Compute attention probs
            attn_probs = F.softmax(attn_scores, dim=-1)

            # Compute attention output
            attn_output = torch.einsum("bns,bnsl->bnsl", attn_probs, V_sj)

            # Scatter the attention output back to the original sequence length
            outputs.scatter_add_(2, torch.arange(sj, seq_len, self.dilation_rate)[None, None, :, None].to(x.device), attn_output)

        # Apply output projection
        outputs = self.out_proj(outputs)

        return outputs
