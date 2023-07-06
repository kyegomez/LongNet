import torch 
import torch.nn as nn
import torch.nn.functional as F
from flash_attn.flash_attention import FlashMHA

# Replace this with your correct GPU device
device = "cuda:0"
dtype=torch.float16

class DilatedAttention(nn.Module):
    def __init__(self, d_model, num_heads, dilation_rate, segment_size):
        super(DilatedAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dilation_rate = dilation_rate
        self.segment_size = segment_size
        self.attention = FlashMHA(embed_dim=d_model, num_heads=num_heads, device=device, dtype=dtype)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Split and sparsify
        x = x.view(batch_size, -1, self.segment_size, self.d_model)
        x = x[:, :, :: self.dilation_rate, :]

        # Perform attention
        attn_output, _ = self.attention(x, x, x)

        # Scatter and concatenate 
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        return attn_output
