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
        casual (optional): If set to True, the attention mechanism is casual. Default: False
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
        casual=False,
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
        self.casual = casual

        self.use_xpos = use_xpos
        self.use_rel_pos_bias = use_rel_pos_bias
        self.qk_norm = qk_norm

        self.attention = FlashAttention(causal=self.casual, dropout=dropout).to(device)

        if use_xpos:
            self.xpos = XPOS(head_dim=dim // heads)
        if use_rel_pos_bias:
            self.relative_bias = RelativePositionBias(
                num_buckets=32, max_distance=128, n_heads=heads
            )

        self.norm = nn.LayerNorm(dim)

        # head offsets
        self.head_offsets = nn.Parameter(torch.randn(heads, dim))

    def get_mask(self, i, j):
        return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 2)

    def forward(self, x):
        print(f"X original shape: {x.shape} and x dtype: {x.dtype}")

        batch_size, seq_len, _ = x.shape
        padding_len = -seq_len % self.segment_size
        x = F.pad(x, (0, 0, 0, padding_len))
        seq_len = seq_len + padding_len
        print(f"Paddex x shape:  {x.shape}")

        if self.use_xpos:
            x = self.xpos(x)

        # Split and sparsify
        x = x.view(batch_size, -1, self.segment_size, self.dim)

        x = x[:, :, :: self.dilation_rate, :]

        # qk_norm
        if self.qk_norm:
            q = x
            k = x
            v = x
            q, k = map(self.norm, (q, k))
        else:
            q = x
            k = x
            v = x

        # Perform attention
        attn_output = self.attention(q, k, v)

        # if use rel pos => apply relative positioning bias
        if self.use_rel_pos_bias:
            attn_output += self.relative_bias(
                batch_size, attn_output.size(1), attn_output.size(1)
            )

        # if casual create a mask and apply to the output
        if self.casual:
            mask = self.get_mask(attn_output.size(1), attn_output.size(1))

            attn_output = attn_output.masked_fill(mask, float("-inf"))

        # apply dropout
        attn_output = self.dropout(attn_output)

        # Scatter and concatenate
        attn_output = attn_output.reshape(batch_size, -1, self.dim)
        return attn_output


class MultiHeadDilatedAttention(DilatedAttention):
    """
    **Description:**
    The `MultiHeadDilatedAttention` class offers a multi-head version of dilated attention, allowing the model to focus on various parts of an input sequence with different patterns.

    **Parameters:**
    - `dim`: Dimension of the model. This is usually a multiple of `heads`.
    - `heads`: Number of attention heads.
    - `dilation_rate`: The rate of dilation for the dilated attention mechanism.
    - `segment_size`: The size of the segment for dilated attention.

    Additionally, the parameters from the parent class `DilatedAttention` are inherited.

    **Methods:**
    - `sparsify(x, sj, head_layer)`: Transforms the tensor by applying the dilated attention offset based on the given head layer. It's used internally and typically not called directly.
    - `forward(x)`: Computes the multi-head dilated attention on the input tensor `x`.
    """

    def __init__(
        self,
        dim,
        heads,
        dilation_rate,
        segment_size,
        *args,
        **kwargs,
    ):
        super(MultiHeadDilatedAttention, self).__init__(
            dim, heads, dilation_rate, segment_size, *args, **kwargs
        )

        self.head_dim = dim // heads

        # define linear layers for Q, K, V projections for each head
        self.query_layers = nn.ModuleList(
            [nn.Linear(dim, self.head_dim) for _ in range(heads)]
        )
        self.key_layers = nn.ModuleList(
            [nn.Linear(dim, self.head_dim) for _ in range(heads)]
        )
        self.value_layers = nn.ModuleList(
            [nn.Linear(dim, self.head_dim) for _ in range(heads)]
        )

        # define linear layer for concat the multi head outputs
        self.concat_layer = nn.Linear(heads * self.head_dim, dim)

    def sparsify(self, x, sj, head_layer):
        x_proj = head_layer(x)
        batch_size, seq_len, _ = x_proj.shape

        padding_len = -seq_len % self.segment_size
        x_proj = F.pad(x_proj, (0, 0, 0, padding_len))
        seq_len += padding_len

        x_proj = x_proj.view(batch_size, -1, self.segment_size, self.head_dim)
        x_proj = x_proj[:, :, sj :: self.dilation_rate, :]

        return x_proj

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        all_heads_output = []

        for j in range(self.heads):
            sj = j % self.dilation_rate

            # create Q, K, V for each head based on offset
            Qi = self.sparsify(x, sj, self.query_layers[j])
            Ki = self.sparsify(x, sj, self.key_layers[j])
            Vi = self.sparsify(x, sj, self.value_layers[j])

            # combine q, k, v tensors for base class forward method
            torch.cat([Qi, Ki, Vi], dim=-1)

            # apply inherited atention on q, k, v
            head_output = super().forward(Qi, Ki, Vi)
            all_heads_output.append(head_output)

        # concat multi head outputs and pdd through final linear layer
        concat_heads = torch.cat(all_heads_output, dim=-1)
        return self.concat_layer(concat_heads)
