import torch 
import torch.nn as nn
import torch.nn.functional as F

from LongNet.utils import XPOS, RelativePositionBias, SparsifyIndices, MixOutputs
from LongNet.attend import FlashAttention
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

        # Prepare sparse indices
        max_subatt_n, sparse_indices, padding_mask = SparsifyIndices(x, [self.segment_size], [self.dilation_rate], self.head_offsets)

        # Split and sparsify
        x = x.view(batch_size, -1, self.segment_size, self.d_model)
        x = x.gather(1, sparse_indices[:, :, :x.size(1)])

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
        attn_output = MixOutputs((batch_size, seq_len, self.d_model), x.dtype, x.device, attn_output, attn_output.sum(dim=-1), sparse_indices)

        return attn_output







import torch
from torch.nn import functional as F

class ConfigurableDilatedAttention(nn.Module):
    def __init__(self, d_model, num_heads, configurations, dropout=0.0, casual=False, use_xpos=False, use_rel_pos_bias=False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.casual = casual
        self.use_xpos = use_xpos
        self.use_rel_pos_bias = use_rel_pos_bias

        # Create a DilatedAttention layer for each configuration
        self.attention_layers = nn.ModuleList([
            DilatedAttention(d_model, num_heads, dilation_rate, segment_size, dropout, casual, use_xpos, use_rel_pos_bias)
            for segment_size, dilation_rate in configurations
        ])

        # Final linear layer
        self.final_linear = nn.Linear(len(configurations) * d_model, d_model)

    def forward(self, x):
        # Calculate maximum padding required
        max_padding = max((segment_size - x.size(1) % segment_size) % segment_size for segment_size, _ in self.configurations)
        x = F.pad(x, (0,0,0,max_padding))

        outputs = []
        for layer, (segment_size, dilation_rate) in zip(self.attention_layers, self.configurations):
            # Fork a new process for each layer
            future = torch.jit.fork(layer, x)
            outputs.append(future)

        # Wait for all processes to finish and gather their outputs
        outputs = [future.wait() for future in outputs]

        # Concatenate all outputs along the feature dimension
        x = torch.cat(outputs, dim=-1)

        # Apply final linear layer
        x = self.final_linear(x)
        
        return x





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
    