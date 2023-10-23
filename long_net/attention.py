import torch
import torch.nn as nn
import torch.nn.functional as F

from long_net.attend import FlashAttention
from long_net.utils import XPOS, RelativePositionBias


# add alibi, qk layer norm, one write head, multihway,
class DilatedAttention(nn.Module):
    """
    Dilated Attention Module.

    Arguments:
        dim: The dimension of the attention layers.
        heads: The number of attention heads.
        dilation_rate: The dilation rate for dilated attention.
        segment_size: The segment size for dilated attention.
        dropout (optional): The dropout probability. Default: 0.0
        causal (optional): If set to True, the attention mechanism is causal. Default: False
        use_xpos (optional): If set to True, xpos is used for positional encoding. Default: False
        use_rel_pos_bias (optional): If set to True, relative position bias is used in the attention mechanism. Default: False

    Usage:
        The `DilatedAttention` class can be used as a module for neural networks and is especially suited for transformer architectures.

        Example:
            attention = DilatedAttention(dim=512, heads=8, dilation_rate=2, segment_size=64, use_xpos=True, use_rel_pos_bias=True)
            output = attention(input_tensor)

        This will return the output tensor after applying dilated attention. The `use_xpos` and `use_rel_pos_bias` parameters allow for switching on positional encoding and relative positional bias respectively.
    """

    def __init__(
        self,
        dim,
        heads,
        dilation_rate,
        segment_size,
        dropout=0.0,
        causal=False,
        use_xpos=False,
        use_rel_pos_bias=False,
        qk_norm=False,
        dtype=torch.float16,
        device="cuda:0",
    ):
        super(DilatedAttention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.dilation_rate = dilation_rate
        self.segment_size = segment_size
        self.dropout = nn.Dropout(dropout)
        self.causal = causal
        self.use_xpos = use_xpos
        self.use_rel_pos_bias = use_rel_pos_bias
        self.qk_norm = qk_norm
        self.dtype = dtype
        self.device = device

        self.attention = FlashAttention(causal=self.causal, dropout=dropout).to(device)

        if use_xpos:
            self.xpos = XPOS(head_dim=dim // heads)
        if use_rel_pos_bias:
            self.relative_bias = RelativePositionBias(
                num_buckets=32, max_distance=128, n_heads=heads
            )

        self.norm = nn.LayerNorm(dim)

        # head offsets
        self.head_offsets = nn.Parameter(torch.randn(heads, dim))

        # Linear Projections
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)

    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        if self.mask is None:
            print('computing mask..')

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        k = 0
        segment_lengths = [4, 8, 16]
        dilation_rates = [1, 2, 4]
        # segment_lengths = [2048, 4096, 8192, 16384, 32768]
        # dilation_rates = [1, 2, 4, 6, 12]
        for i in range(len(mask)):
            for j in range(len(mask[0])):
                will_mask = True
                for segment_length, dilation_rate in zip(segment_lengths, dilation_rates):
                    if np.floor(i/segment_length) == np.floor(j/segment_length) and i % dilation_rate == 0 and j % dilation_rate == 0:
                        will_mask = False
                if will_mask:
                    mask[i][j] = True
                k += 1
        self.register_buffer("mask", mask, persistent=False)
        self.mask = mask
        return mask

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        padding_len = -seq_len % self.segment_size
        x = F.pad(x, (0, 0, 0, padding_len))
        seq_len = seq_len + padding_len

        if self.use_xpos:
            x = self.xpos(x)

        # Split and sparsify
        x = x.view(batch_size, -1, self.segment_size, self.dim)
        x = x[:, :, :: self.dilation_rate, :]

        # qk_norm
        if self.qk_norm:
            q, k, v = map(self.norm, (self.proj_q(x), self.proj_k(x), self.proj_v(x)))
        else:
            q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)

        # if self.qk_norm:
        #     q = self.proj(x).transpose(-2, -3)
        #     k = self.proj(x).transpose(-2, -3)
        #     v = self.proj(x).transpose(-2, -3)
        #     q, k = map(self.norm, (q, k))
        #     q, k, v = q.tranposse(-2, -3), k.tranpose(-2, -3), v.transpose(-2, -3)
        # else:
        #     q = self.proj_q(x)
        #     k = self.proj_k(x)
        #     v = self.proj_v(x)

        # Perform attention
        attn_output = self.attention(q, k, v)

        # if use rel pos => apply relative positioning bias
        if self.use_rel_pos_bias:
            attn_output += self.relative_bias(
                batch_size, attn_output.size(1), attn_output.size(1)
            )

        # if causal create a mask and apply to the output
        if self.causal:
            mask = self.get_mask(n=attn_output.size(1), device='cuda:0')

            attn_output = attn_output.masked_fill(mask, float("-inf"))

        # apply dropout
        attn_output = self.dropout(attn_output)
        # Scatter and concatenate
        attn_output = attn_output.reshape(batch_size, -1, self.dim)
        return attn_output
