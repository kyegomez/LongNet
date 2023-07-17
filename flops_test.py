from LongNet.attention import DilatedAttention

import torch
from pthflops import count_ops
import time
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bsz = 32
d_model = 512
num_heads = 8
head_dim = d_model // num_heads
hidden_size = d_model
sequence_lengths = list(range(500, 2500, 500))

time_taken = []
tflops_per_s = []

model = DilatedAttention(num_heads, head_dim, hidden_size).to(device)

for seq_len in sequence_lengths:
    x = torch.randn(bsz, seq_len, d_model).to(device)

    start_time = time.time()
    output = model(x, x, x)
    end_time = time.time()

    # Count operations on a single forward pass
    count, _ = count_ops(model, x)

    # Convert to TFLOPs
    tflops = count / 1e12

    # Calculate TFLOPs/s
    tflops_per_s.append(tflops / (end_time - start_time))
    time_taken.append(end_time - start_time)

# Plotting
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(sequence_lengths, time_taken)
plt.title('Time Taken vs Sequence Length')
plt.xlabel('Sequence Length')
plt.ylabel('Time Taken (s)')

plt.subplot(1,2,2)
plt.plot(sequence_lengths, tflops_per_s)
plt.title('Performance vs Sequence Length')
plt.xlabel('Sequence Length')
plt.ylabel('Performance (TFLOPs/s)')

plt.tight_layout()
plt.show()