import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange


class DilatedAttention(nn.Module):
    def __init__(
        self,
        dim,
        segment_length,
        dilated_rate,
        dropout=0.1,
        qk_norm=True,
        use_xpos=True,
    ):
        super(DilatedAttention, self).__init__()
        self.segment_length = segment_length
        self.dilated_rate = dilated_rate
        self.qk_norm = qk_norm
        self.use_xpos = use_xpos

        self.dropout = nn.Dropout(dropout)

        self.to_keys = nn.Linear(dim, dim, bias=False)
        self.to_queries = nn.Linear(dim, dim, bias=False)
        self.to_values = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

        # softmax denominator
        self.softmax = nn.Softmax(dim=-1)

        # if use_xpos:
        #     self.xpos = XPOS(head_dim = dim // heads)

    def _segment_and_sparsify(self, x):
        # Divide x into segments
        x = rearrange(x, "b (l s) d -> b l s d", s=self.segment_length)

        # Sparsify
        x = x[:, :, :: self.dilated_rate]

        # Flatten the segmented and sparsified tensor
        x = rearrange(x, "b l s d -> b (l s) d")

        return x

    def forward(self, Q, K, V):
        Q, K, V = map(self._segment_and_sparsify, (Q, K, V))

        Q, K, V = self.to_queries(Q), self.to_keys(K), self.to_values(V)

        # norm qk values
        if self.qk_norm:
            Q, K = map(self.norm, (Q, K))
        else:
            pass

        # # Apply self attention on sparsified segments, use F.scaled_dot product in future
        # attn_weights = F.softmax(Q @ K.transpose(-2, -1), dim=-1)
        # attn_out = attn_weights @ V
        # return attn_out

        # flash
        with torch.backends.cuda.sdp_kernel(enable_math=True):
            out = F.scaled_dot_product_attention(Q, K, V)

            # dropout =
            out = self.dropout(out)

            # softmax
            # out = self.softmax(out)
        return out


# Define input dimensions
BATCH_SIZE = 8
SEQUENCE_LENGTH = 32
DIM = 64
SEGMENT_LENGTH = 4
DILATED_RATE = 2

# Create random input tensors
Q = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, DIM)
K = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, DIM)
V = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, DIM)

# Create the DilatedAttention model
model = DilatedAttention(DIM, SEGMENT_LENGTH, DILATED_RATE)

# Forward pass
output = model(Q, K, V)
print(output)
