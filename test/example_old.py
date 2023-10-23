import timeit
import torch
from long_net.attention import DilatedAttention

# model config
dim = 512
heads = 8
dilation_rate = 2
segment_size = 64

device = "cuda:0"
dtype = torch.float16

# input data
batch_size = 32
seq_len = 1024


# create model and data
model = DilatedAttention(dim, heads, dilation_rate, segment_size).to(device)
x = torch.randn((batch_size, seq_len, dim), device=device, dtype=dtype)


# test forward pass
with torch.no_grad():
    output = model(x)
    print(f"Output shape: {output.shape}")  # expected (batch_size, seq_Len)


# benchmark model
num_runs = 1000
start_time = timeit.default_timer()
for _ in range(num_runs):
    model(x)

elapsed_time = timeit.default_timer() - start_time
print(f"Average forward pass time: {elapsed_time / num_runs:.6f} seconds")
