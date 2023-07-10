import torch 
import torch.nn as nn
import torch.nn.functional as F

from torchscale.component.xpos_relative_position import XPOS
from torchscale.component.relative_position_bias import RelativePositionBias

# from LongNet.attend import FlashMHA
from flash_attn.flash_attn.flash_attention import FlashMHA

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







#second iteration the weighted sum of the different dilated + offsets for the different heads
class DilatedAttention(nn.Module):
    def __init__(self, d_model, num_heads, dilation_rate, segment_size, dropout=0.0, casual=False, use_xpos=False, use_rel_pos_bias=False):
        super(DilatedAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.dilation_rate = dilation_rate
        self.segment_size = segment_size

        self.attentions = nn.ModuleList([FlashMHA(embed_dim=d_model, num_heads=num_heads, device=device, dtype=dtype) for _ in range(self.dilation_rate)])
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

        #collect outputs from each attention head
        all_head_outputs = []
        for head_idx, attention in enumerate(self.attentions):
            offset = head_idx % self.dilation_rate

            x_ = x[:, offset::self.dilation_rate, :]  # Apply offset for each head
            x_ = x_.contiguous().view(batch_size, -1, self.segment_size, self.d_model)

            attn_output, _ = attention(x_, x_, x_)
            if self.use_rel_pos_bias:
                attn_output += self.relative_bias(batch_size, attn_output.size(1), attn_output.size(1))

            if self.casual:
                mask = self.get_mask(attn_output.size(1), attn_output.size(1))
                attn_output = attn_output.masked_fill(mask, float('-inf'))

            attn_output = self.dropout(attn_output)

            #resize back to original size
            attn_output_resized = torch.zeros((batch_size, seq_len, self.d_model), device=device, dtype=dtype)
            attn_output_resized[:, offset::self.dilation_rate, :] = attn_output.contiguous().view(batch_size, -1, self.d_model)
            
            all_head_outputs.append(attn_output_resized)

        #concatenate the outputs of different heads
        outputs_concatenated = torch.cat(all_head_outputs, dim=-1)

        return outputs_concatenated





# # from flash_attn.flash_blocksparse_attention import FlashBlocksparseMHA
# #perhaps integrate integrate dynamic sparse attention
# class BlocksparseDilatedAttention(nn.Module):
#     def __init__(self, d_model, num_heads, dilation_rate, segment_size, sparsity_config, dropout=0.0, causal=False, use_xpos=False, use_rel_pos_bias=False):
#         super(BlocksparseDilatedAttention, self).__init__()
#         self.d_model = d_model
#         self.num_heads = num_heads

#         self.dilation_rate = dilation_rate
#         self.segment_size = segment_size

#         self.attentions = nn.ModuleList([FlashBlocksparseMHA(embed_dim=d_model, num_heads=num_heads, sparsity_config=sparsity_config, device=device, dtype=dtype) for _ in range(self.dilation_rate)])
#         self.dropout = nn.Dropout(dropout)
#         self.causal = causal

#         self.use_xpos = use_xpos
#         self.use_rel_pos_bias = use_rel_pos_bias

#         if use_xpos:
#             self.xpos = XPOS(head_dim=d_model//num_heads)
#         if use_rel_pos_bias:
#             self.relative_bias = RelativePositionBias(num_buckets=32, max_distance=128, n_heads=num_heads)

#     def get_mask(self, i, j):
#         return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 2)

#     def forward(self, x):
#         batch_size, seq_len, _ = x.shape

#         if self.use_xpos:
#             x = self.xpos(x)

#         # Collect outputs from each attention head
#         all_head_outputs = []
#         for head_idx, attention in enumerate(self.attentions):
#             offset = head_idx % self.dilation_rate

#             x_ = x[:, offset::self.dilation_rate, :]  # Apply offset for each head
#             x_ = x_.contiguous().view(batch_size, -1, self.segment_size, self.d_model)

#             attn_output, _ = attention(x_, x_, x_)
#             if self.use_rel_pos_bias:
#                 attn_output += self.relative_bias(batch_size, attn_output.size(1), attn_output.size(1))

#             if self.causal:
#                 mask = self.get_mask(attn_output.size(1), attn_output.size(1))
#                 attn_output = attn_output.masked_fill(mask, float('-inf'))

#             attn_output = self.dropout(attn_output)

#             # Resize back to original size
#             attn_output_resized = torch.zeros((batch_size, seq_len, self.d_model), device=device, dtype=dtype)
#             attn_output_resized[:, offset::self.dilation_rate, :] = attn_output.contiguous().view(batch_size, -1, self.d_model)
            
#             all_head_outputs.append(attn_output_resized)

#         # Concatenate the outputs of different heads
#         outputs_concatenated = torch.cat(all_head_outputs, dim=-1)

#         return outputs_concatenated


#distributed dilated attention based on second iteration
import torch.distributed as dist

class DistributedDilatedAttention(nn.Module):
    def __init__(self, d_model, num_heads, dilation_rate, segment_size, dropout=0.0, casual=False, use_xpos=False, use_rel_pos_bias=False):
        super(DistributedDilatedAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.dilation_rate = dilation_rate
        self.segment_size = segment_size

        self.attentions = nn.ModuleList([FlashMHA(embed_dim=d_model, num_heads=num_heads, device=device) for _ in range(self.dilation_rate)])
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

        # Collect outputs from each attention head
        all_head_outputs = []
        for head_idx, attention in enumerate(self.attentions):
            offset = head_idx % self.dilation_rate

            x_ = x[:, offset::self.dilation_rate, :]  # Apply offset for each head
            x_ = x_.contiguous().view(batch_size, -1, self.segment_size, self.d_model)

            # compute attention locally, gather the key-value pairs before computing the attention
            attn_output, _ = attention(x_, x_, x_)
            dist.all_gather(attn_output, attn_output)

            if self.use_rel_pos_bias:
                attn_output += self.relative_bias(batch_size, attn_output.size(1), attn_output.size(1))

            if self.casual:
                mask = self.get_mask(attn_output.size(1), attn_output.size(1))
                attn_output = attn_output.masked_fill(mask, float('-inf'))

            attn_output = self.dropout(attn_output)

            # Resize back to original size
            attn_output_resized = torch.zeros((batch_size, seq_len, self.d_model), device=device)
            attn_output_resized[:, offset::self.dilation_rate, :] = attn_output.contiguous().view(batch_size, -1, self.d_model)
            
            all_head_outputs.append(attn_output_resized)

        # Concatenate the outputs of different heads
        outputs_concatenated = torch.cat(all_head_outputs, dim=-1)

        return outputs_concatenated







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
    

# class DilatedAttention(nn.Module):
#     def __init__(self, d_model, num_heads, dilation_rate, segment_size, dropout=0.0, casual=False, use_xpos=False, use_rel_pos_bias=False):
#         super(DilatedAttention, self).__init__()
#         self.d_model = d_model
#         self.num_heads = num_heads

#         self.dilation_rate = dilation_rate
#         self.segment_size = segment_size

#         self.attentions = nn.ModuleList([FlashMHA(embed_dim=d_model, num_heads=num_heads, device=device, dtype=dtype) for _ in range(self.dilation_rate)])
#         self.dropout = nn.Dropout(dropout)
#         self.casual = casual

#         self.use_xpos = use_xpos
#         self.use_rel_pos_bias = use_rel_pos_bias

#         if use_xpos:
#             self.xpos = XPOS(head_dim=d_model//num_heads)
#         if use_rel_pos_bias:
#             self.relative_bias = RelativePositionBias(num_buckets=32, max_distance=128, n_heads=num_heads)

#     def get_mask(self, i, j):
#         return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 2)

#     def forward(self, x):
#         batch_size, seq_len, _ = x.shape

#         if self.use_xpos:
#             x = self.xpos(x)

#         x = x.view(batch_size, -1, self.segment_size, self.d_model)
#         x = x[:, :, :: self.dilation_rate, :]

#         # collect output from each attention head
#         attn_outputs = []
#         for attention in self.attentions:
#             attn_output, _ = attention(x, x, x)
#             if self.use_rel_pos_bias:
#                 attn_output += self.relative_bias(batch_size, attn_output.size(1), attn_output.size(1))

#             if self.casual:
#                 mask = self.get_mask(attn_output.size(1), attn_output.size(1))
#                 attn_output = attn_output.masked_fill(mask, float('-inf'))

#             attn_output = self.dropout(attn_output)
#             attn_outputs.append(attn_output)

#         # apply weighted sum and softmax
#         attn_outputs = torch.stack(attn_outputs, dim=0)
#         attn_outputs = F.softmax(attn_outputs, dim=0)
#         attn_outputs = torch.sum(attn_outputs, dim=0)

#         attn_outputs = attn_outputs.view(batch_size, -1, self.d_model)

#         return attn_outputs
