import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention # switch for flash multi head attention

class DilatedAttenion(nn.Module):
    def __init__(self, d_model, num_heads, dilation_rate, segment_size):
        super(DilatedAttenion, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dilation_rate = dilation_rate
        self.segment_size = segment_size
        self.attention = MultiheadAttention(embed_dim=d_model, num_heads=num_heads)


    def forward(self, x):
        batch_size, seq_len, _ = x.shape


        #split and sparsify
        x = x.view(batch_size, -1, self.segment_size, self.d_model)
        x = x[:, :, :: self.dilation_rate, :]

        #perform attentioon
        attn_output, _ = self.attention(x, x, x)

        #scatter and concatenate 
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        return attn_output
