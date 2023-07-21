import torch
import torch.nn.functional as F
import time

from LongNet.attention import DilatedAttention


# Initialize parameters
bsz = 32
d_model = 512
num_heads = 8
dilation_rate = 2
segment_size = 512  # You might want to adjust this
dropout = 0.1
casual = False
use_xpos = False
use_rel_pos_bias = False

sequence_lengths = list(range(500, 2500, 500))

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype=torch.float32

# Initialize model
model = DilatedAttention(
    d_model=d_model, 
    num_heads=num_heads, 
    dilation_rate=dilation_rate, 
    segment_size=segment_size, 
    dropout=dropout, 
    casual=casual, 
    use_xpos=use_xpos, 
    use_rel_pos_bias=use_rel_pos_bias
).to(device)

time_taken = []
tflops_per_s = []

# Benchmark model
for seq_len in sequence_lengths:
    x = torch.randn(bsz, seq_len, d_model).to(device).type(dtype)
    torch.cuda.synchronize()

    start = time.time()
    output = model(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    time_taken.append(elapsed)
    total_flops = 4 * seq_len**2 * (d_model // num_heads) * num_heads
    tflops_per_s.append(total_flops / elapsed / 1e12)  # Convert to TFLOPs

# Print benchmark results
for seq_len, elapsed, tflops in zip(sequence_lengths, time_taken, tflops_per_s):
    print(f"Sequence length: {seq_len}, Time elapsed: {elapsed} s, TFLOPs/s: {tflops}")


# # Plotting
# plt.figure(figsize=(10,4))

# plt.subplot(1,2,1)
# plt.plot(sequence_lengths, time_taken)
# plt.title('Time Taken vs Sequence Length')
# plt.xlabel('Sequence Length')
# plt.ylabel('Time Taken (s)')

# plt.subplot(1,2,2)
# plt.plot(sequence_lengths, tflops_per_s)
# plt.title('Performance vs Sequence Length')
# plt.xlabel('Sequence Length')
# plt.ylabel('Performance (TFLOPs/s)')

# plt.tight_layout()
# plt.show()
