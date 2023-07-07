import torch 
import torch.nn as nn
import torch.nn.functional as F
from flash_attn.flash_attention import FlashMHA

# Replace this with your correct GPU device
device = "cuda:0"
dtype=torch.float16

class DilatedAttention(nn.Module):
    def __init__(self, d_model, num_heads, dilation_rate, segment_size, dropout=0.0):
        super(DilatedAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dilation_rate = dilation_rate
        self.segment_size = segment_size
        self.attention = FlashMHA(embed_dim=d_model, num_heads=num_heads, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Split and sparsify
        x = x.view(batch_size, -1, self.segment_size, self.d_model)
        x = x[:, :, :: self.dilation_rate, :]

        # Perform attention
        attn_output, _ = self.attention(x, x, x)

        #apply dropout
        attn_output = self.dropout(attn_output)

        # Scatter and concatenate 
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        return attn_output



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