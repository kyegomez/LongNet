import torch 
import torch.nn as nn
import torch.nn.functional as F

from LongNet.utils import XPOS, RelativePositionBias
from LongNet.attention import FlashMHA

# Replace this with your correct GPU device
device = "cuda:0"
dtype=torch.float16





# from flash_attn.flash_blocksparse_attention import FlashBlocksparseMHA
#perhaps integrate integrate dynamic sparse attention
class BlocksparseDilatedAttention(nn.Module):
    def __init__(self, d_model, num_heads, dilation_rate, segment_size, sparsity_config, dropout=0.0, causal=False, use_xpos=False, use_rel_pos_bias=False):
        super(BlocksparseDilatedAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.dilation_rate = dilation_rate
        self.segment_size = segment_size

        self.attentions = nn.ModuleList([FlashBlocksparseMHA(embed_dim=d_model, num_heads=num_heads, sparsity_config=sparsity_config, device=device, dtype=dtype) for _ in range(self.dilation_rate)])
        self.dropout = nn.Dropout(dropout)
        self.causal = causal

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

            attn_output, _ = attention(x_, x_, x_)
            if self.use_rel_pos_bias:
                attn_output += self.relative_bias(batch_size, attn_output.size(1), attn_output.size(1))

            if self.causal:
                mask = self.get_mask(attn_output.size(1), attn_output.size(1))
                attn_output = attn_output.masked_fill(mask, float('-inf'))

            attn_output = self.dropout(attn_output)

            # Resize back to original size
            attn_output_resized = torch.zeros((batch_size, seq_len, self.d_model), device=device, dtype=dtype)
            attn_output_resized[:, offset::self.dilation_rate, :] = attn_output.contiguous().view(batch_size, -1, self.d_model)
            
            all_head_outputs.append(attn_output_resized)

        # Concatenate the outputs of different heads
        outputs_concatenated = torch.cat(all_head_outputs, dim=-1)

        return outputs_concatenated



