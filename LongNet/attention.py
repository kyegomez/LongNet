import torch
import torch.nn as nn
import torch.nn.functional as F

from longnet.attend import FlashAttention
from longnet.utils import XPOS, RelativePositionBias

device = "cuda:0"
dtype = torch.float16


class ParallelWrapper:
    """
    A simple wrapper to enable easy usage of data parallelism.

    Arguments:
        model: The neural network model to be parallelized.
        device (optional): The device to which the model should be moved. Default: "cuda".
        use_data_parallel (optional): A boolean flag to indicate whether to use data parallelism or not. Default: True.

    Usage:
        The `ParallelWrapper` class can be used as a wrapper for neural networks and is especially suited for transformer architectures.

        Example:
            model = nn.Linear(512, 512)
            model = ParallelWrapper(model)

        This will return the model wrapped in the `ParallelWrapper` class. The `use_data_parallel` parameter allows for switching on data parallelism.
    """

    def __init__(self, model, device="cuda", use_data_parallel=True):
        self.model = model.to(device)
        self.use_data_parallel = use_data_parallel
        self.device = device

        if self.use_data_parallel and torch.cuda.device_count() < 1:
            print(f"Using {torch.cuda.device_count()} GPUS")
            self.model = nn.DataParallel(self.model)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        return self

    def __getattr__(self, name):
        # redirect attribute access to the internal model to allow direct access to its methods and props
        return getattr(self.model, name)


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
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)

    def get_mask(self, i, j):
        return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 2)

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

        # Perform attention
        attn_output = self.attention(q, k, v)

        # if use rel pos => apply relative positioning bias
        if self.use_rel_pos_bias:
            attn_output += self.relative_bias(
                batch_size, attn_output.size(1), attn_output.size(1)
            )

        # if causal create a mask and apply to the output
        if self.causal:
            mask = self.get_mask(attn_output.size(1), attn_output.size(1))

            attn_output = attn_output.masked_fill(mask, float("-inf"))

        # apply dropout
        attn_output = self.dropout(attn_output)
        # Scatter and concatenate
        attn_output = attn_output.reshape(batch_size, -1, self.dim)
        return attn_output

