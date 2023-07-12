
import torch 
import torch.nn as nn
import torch.nn.functional as F

from LongNet.utils import XPOS, RelativePositionBias
from LongNet.attention import FlashMHA

# Replace this with your correct GPU device
device = "cuda:0"
dtype=torch.float16




#add alibi, qk layer norm, one write head, multihway, 
class DilatedAttentionold(nn.Module):
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
        super(DilatedAttentionold, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.dilation_rate = dilation_rate
        self.segment_size = segment_size

        self.attention = FlashMHA(embed_dim=d_model, num_heads=num_heads, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        self.casual = casual

        self.use_xpos = use_xpos
        self.use_rel_pos_bias = use_rel_pos_bias

        if use_xpos:
            self.xpos = XPOS(head_dim=d_model//num_heads)
        if use_rel_pos_bias:
            self.relative_bias = RelativePositionBias(num_buckets=32, max_distance=128, n_heads=num_heads)

        #head offsets
        self.head_offsets = nn.Parameter(torch.randn(num_heads, d_model))

    def get_mask(self, i, j):
        return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 2)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        if self.use_xpos:
            x = self.xpos(x)
        
        # Split and sparsify
        x = x.view(batch_size, -1, self.segment_size, self.d_model)
        x = x[:, :, :: self.dilation_rate, :]

        # Perform attention
        attn_output, _ = self.attention(x, x, x)

        #if use rel pos => apply relative positioning bias 
        if self.use_rel_pos_bias:
            attn_output += self.relative_bias(batch_size, attn_output.size(1), attn_output.size(1))

        # if casual create a mask and apply to the output
        if self.casual:
            mask = self.get_mask(attn_output.size(1), attn_output.size(1))
            attn_output = attn_output.masked_fill(mask, float('-inf'))

        # apply dropout
        attn_output = self.dropout(attn_output)

        # Scatter and concatenate 
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        return attn_output
