import timeit
import torch
from longnet.attention import MultiHeadDilatedAttention

# Model config
d_model = 512
num_heads = 8
dilation_rate = 2
segment_size = 64

device = "cuda:0"
dtype = torch.float16

# Input data
batch_size = 32
seq_len = 8192

# Create model and data
# Convert model to dtype along with moving to device
model = (
    MultiHeadDilatedAttention(d_model, num_heads, dilation_rate, segment_size)
    .to(device)
    .to(dtype)
)
x = torch.randn((batch_size, seq_len, d_model), device=device, dtype=dtype)

# Test forward pass
with torch.no_grad():
    output = model(x)
    print(f"Output shape: {output.shape}")  # Expected (batch_size, seq_len)

# Benchmark model
num_runs = 1000
start_time = timeit.default_timer()
for _ in range(num_runs):
    model(x)

elapsed_time = timeit.default_timer() - start_time
print(f"Average forward pass time: {elapsed_time / num_runs:.6f} seconds")
