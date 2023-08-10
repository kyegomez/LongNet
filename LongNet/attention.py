import torch
import torch.nn as nn
import torch.nn.functional as F

from LongNet.attend import FlashAttention
from LongNet.utils import XPOS, RelativePositionBias, SparsifyIndices

from typing import Tuple, Union

device = "cuda:0"
dtype=torch.float16


def mix_outputs(
    out_shape: Tuple[int, int, int],
    out_dtype: torch.dtype,
    out_device: Union[torch.device, str],
    a_os: torch.Tensor,
    a_denoms: torch.Tensor,
    a_indices: torch.Tensor,
) -> torch.Tensor:
    # calculate sums of softmax denominators
    att_denom_sums = torch.zeros((out_shape[0], out_shape[1]), device=out_device)
    att_denom_sums.scatter_add_(1, a_indices[:, :, 0], a_denoms)

    # select attention softmax denominator sums for current sparse indices
    sparse_att_denom_sum = torch.gather(att_denom_sums, 1, a_indices[:, :, 0])

    # compute alphas
    alphas = torch.divide(a_denoms, sparse_att_denom_sum)[:, :, None]

    out = torch.zeros(out_shape, dtype=out_dtype, device=out_device)

    out.scatter_add_(
        1,
        a_indices[:, :, :out.shape[2]],
        torch.multiply(a_os, alphas),
    )

    return out


#add alibi, qk layer norm, one write head, multiway, 
class DilatedAttention(nn.Module):
    """
    Dilated Attention Module.

    Arguments:
        d_model: The dimension of the attention layers.
        num_heads: The number of attention heads.
        dilation_rate: The dilation rate for dilated attention.
        segment_size: The segment size for dilated attention.
        dropout (optional): The dropout probability. Default: 0.0
        casual (optional): If set to True, the attention mechanism is casual. Default: False
        use_xpos (optional): If set to True, xpos is used for positional encoding. Default: False
        use_rel_pos_bias (optional): If set to True, relative position bias is used in the attention mechanism. Default: False

    Usage:
        The `DilatedAttention` class can be used as a module for neural networks and is especially suited for transformer architectures.

        Example:
            attention = DilatedAttention(d_model=512, num_heads=8, dilation_rate=2, segment_size=64, use_xpos=True, use_rel_pos_bias=True)
            output = attention(input_tensor)

        This will return the output tensor after applying dilated attention. The `use_xpos` and `use_rel_pos_bias` parameters allow for switching on positional encoding and relative positional bias respectively.
    """
    def __init__(self, d_model, num_heads, dilation_rate, segment_size, dropout=0.0, casual=False, use_xpos=False, use_rel_pos_bias=False):
        super(DilatedAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.dilation_rate = dilation_rate
        self.segment_size = segment_size

        self.dropout = nn.Dropout(dropout)
        self.casual = casual

        self.use_xpos = use_xpos
        self.use_rel_pos_bias = use_rel_pos_bias

        self.attention = FlashAttention(causal=self.casual, dropout=dropout).to(device)

        if use_xpos:
            self.xpos = XPOS(head_dim=d_model//num_heads)
        if use_rel_pos_bias:
            self.relative_bias = RelativePositionBias(num_buckets=32, max_distance=128, n_heads=num_heads)

        #head offsets
        self.head_offsets = nn.Parameter(torch.randn(num_heads, d_model))

    def get_mask(self, i, j):
        return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 2)

    def forward(self, x):
        # get dimensions
        batch_size, seq_len, _ = x.shape

        # calculate the necessary padding
        padding_len = -seq_len % self.segment_size
        x = F.pad(x, (0,0,0,padding_len))
        seq_len = seq_len + padding_len

        if self.use_xpos:
            x = self.xpos(x)


        head_idx = int(self.head_offsets[0, 0].item())

        # Prepare sparse indices
        # max_subatt_n, sparse_indices, padding_mask = sparsify_indices(x, [self.segment_size], [self.dilation_rate], self.head_offsets)
        max_subatt_n, sparse_indices, padding_mask = SparsifyIndices(x, [self.segment_size], [self.dilation_rate], head_idx)

        # Split and sparsify
        x = x.view(batch_size, -1, self.segment_size, self.d_model)

        #Gather operation
        x_dim1 = x.size(1)
        x = x.gather(2, sparse_indices[:, :x_dim1, :].unsqueeze(-1).expand(-1, -1, -1, self.d_model))


        # Perform attention
        attn_output = self.attention(x, x, x)

        #if use rel pos => apply relative positioning bias 
        if self.use_rel_pos_bias:
            attn_output += self.relative_bias(batch_size, attn_output.size(1), attn_output.size(1))

        # if casual create a mask and apply to the output
        if self.casual:
            mask = self.get_mask(attn_output.size(1), attn_output.size(1))
            attn_output = attn_output.masked_fill(mask, float('-inf'))

        # apply dropout
        attn_output = self.dropout(attn_output)

        # Mix outputs
        attn_output = mix_outputs((batch_size, seq_len, self.d_model), x.dtype, x.device, attn_output, attn_output.sum(dim=-1), sparse_indices)

        return attn_output











class MultiHeadDilatedAttention:
    def __init__(self, d_model, num_heads, segment_size, dilation_rate, dropout=0.0, casual=False, use_xpos=False, use_rel_pos_bias=False):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.segment_size = segment_size
        self.dilation_rate = dilation_rate
        self.head_dim = d_model // num_heads

        assert (self.head_dim * num_heads == d_model), 'Embedding dimebsion should be divisible by number of heads'

        self.dilated_attention = DilatedAttention(d_model, num_heads, dilation_rate, segment_size, dropout, casual, use_xpos, use_rel_pos_bias)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        #calculate the necessaary padding
        padding_len = -seq_len % self.segment_size
        x = F.pad(x, (0, 0, 0, padding_len))

        #init output tensor
        outputs = torch.zeros_like(x)

        #perform dilated attention on each head
        outputs = self.dilated_attention(x)

        return outputs



