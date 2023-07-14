import torch 
import torch.nn as nn
import torch.nn.functional as F

from LongNet.utils import XPOS, RelativePositionBias
from LongNet.attention import FlashMHA

device = "cuda:0"
dtype=torch.float16


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

        self.softmax = nn.Softmax(dim=-1)

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

        #calculate the weights for the different dilated attentions
        weights = self.softmax(torch.tensor([1.0 / self.dilation_rate for _ in range(self.dilation_rate)], device=device, dtype=dtype))

        #apply the weights to the outputs of the different heads
        outputs_weighted = sum(w * out for w, out in zip(weights, all_head_outputs))

        return outputs_weighted




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