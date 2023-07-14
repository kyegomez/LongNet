import timeit
import torch
from LongNet.iterations.DilatedAttentionOld import DilatedAttentionold as DilatedAttention
import time

# Define sequence lengths to test
seq_lengths = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

# Define batch size and feature dimension
batch_size = 32
d_model = 512

device = 'cuda:0'

# Initialize DilatedAttentionold module
attention = DilatedAttention(d_model=d_model, num_heads=8, dilation_rate=2, segment_size=64, use_xpos=False, use_rel_pos_bias=False)

# Move the model to GPU
attention.to(device)

# Benchmark each sequence length
for seq_len in seq_lengths:
    # Create a random input tensor
    x = torch.randn(batch_size, seq_len, d_model).to(device)

    # Warm up GPU
    for _ in range(10):
        _ = attention(x)

    # Benchmark
    start_time = time.time()
    for _ in range(100):
        _ = attention(x)
    end_time = time.time()

    # Calculate average forward pass time
    avg_time = (end_time - start_time) / 100

    print(f"Sequence length: {seq_len}, Average forward pass time: {avg_time} seconds")