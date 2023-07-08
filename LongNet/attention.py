import torch 
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import FlashMHA

from torchscale import XPOS, RelativePositionBias

# Replace this with your correct GPU device
device = "cuda:0"
dtype=torch.float16

#add alibi, qk layer norm, one write head, multihway, 
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

        self.attention = FlashMHA(embed_dim=d_model, num_heads=num_heads, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        self.casual = casual

        self.use_xpos = use_xpos
        self.use_rel_pos_bias = use_rel_pos_bias

        if use_xpos:
            self.xpos = XPOS(head_dim=d_model//num_heads)
        if use_rel_pos_bias:
            self.relative_bias = RelativePositionBias(num_buckets=32, max_distance=128, n_heads=num_heads)

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



class MultiModalDilationAttention(nn.Module):
    def __init__(self, d_model, num_heads, dilation_rate, segment_size, dropout=0.0, casual=False, num_modalities=2):
        super(MultiModalDilationAttention, self).__init__()

        self.d_model = d_model
        self.num_modalities = num_modalities
        self.dilated_attns = nn.ModuleList(
            [DilatedAttention(d_model, num_heads, dilation_rate, segment_size, dropout, casual) for _ in range(num_modalities)]
        )
        self.cross_modality_attn = DilatedAttention(num_modalities * d_model, num_heads, dilation_rate, segment_size, dropout, casual)

    def forward(self, x):
        modality_outputs = []
        for modality_data, attn in zip(x, self.dilated_attns):
            modality_outputs.append(attn(modality_data))
        
        cross_modality_input = torch.cat(modality_outputs, dim=-1)
        cross_modality_output = self.cross_modality_attn_(cross_modality_input)

        return cross_modality_output

class LongNetTransformer(nn.Module):
    def __init__(self, d_model, num_heads, dilation_rates, segment_sizes):
        super(LongNetTransformer, self).__init__()
        assert len(dilation_rates) == len(segment_sizes), "dilation_rates and segment_sizes should have the same length"


        self.d_model = d_model
        self.num_heads = num_heads
        self.dilation_rates = dilation_rates
        self.segment_sizes = segment_sizes
        
        self.dilated_attention_layers = nn.ModuleList(
            [DilatedAttention(d_model, num_heads, dilation_rate, segment_size)]
            for dilation_rate, segment_size in zip(dilation_rates, segment_sizes)
        )

    def forward(self, x):
        #accumlate outputs from different layers
        outputs = []

        #process each dilated attention layer
        for i in range(len(self.dilated_attention_layers)):
            output = self.dilated_attention_layers[i](x)
            outputs.append(output)

        #combine the outputs
        output = torch.sum(torch.stack(outputs), dim=0)

        return output