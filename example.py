import torch
from long_net.attention import DilatedAttention


# model config
dim = 512
heads = 8
dilation_rate = 2
segment_size = 64

# input data
batch_size = 32
seq_len = 8192


# create model and data
model = DilatedAttention(dim, heads, dilation_rate, segment_size)
x = torch.randn((batch_size, seq_len, dim))

output = model(x)
print(output)
