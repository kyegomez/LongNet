import torch
import time

from longnet.attention import DilatedAttention
import matplotlib.pyplot as plt



# Define sequence lengths to test
seq_lengths = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 64000]

# Define batch size and feature dimension
batch_size = 32
d_model = 512

device = 'cuda:0'

# Initialize DilatedAttentionold module
attention = DilatedAttention(d_model=d_model, num_heads=8, dilation_rate=2, segment_size=64, use_xpos=False, use_rel_pos_bias=False)

# Move the model to GPU
attention.to(device)

# Prepare a list to store times
times = []

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

    # Store the time
    times.append(avg_time)

    print(f"Sequence length: {seq_len}, Average forward pass time: {avg_time} seconds")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(seq_lengths, times, marker='o')
plt.title('Average forward pass time for different sequence lengths')
plt.xlabel('Sequence length')
plt.ylabel('Average forward pass time (seconds)')
plt.grid(True)
plt.show()