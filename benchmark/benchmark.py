import time
import torch
import matplotlib.pyplot as plt
from long_net.attention import DilatedAttention
from long_net.attend import FlashAttention


class DilatedAttentionTest:
    def __init__(self, batch_size, dim, device):
        self.model = DilatedAttention(
            dim=dim,
            heads=8,
            dilation_rate=2,
            segment_size=64,
            use_xpos=False,
            use_rel_pos_bias=False,
        )
        self.model.to(device)
        self.batch_size = batch_size
        self.device = device

    def test(self, seq_len):
        x = torch.randn(self.batch_size, seq_len, self.model.dim).to(self.device)

        # warm up gpu
        for _ in range(10):
            _ = self.model(x)

        # benchmark
        start_time = time.time()
        for _ in range(100):
            _ = self.model(x)
        end_time = time.time()

        # calculate average forward pass time
        avg_time = (end_time - start_time) / 100

        return avg_time


class FlashAttentionTest(DilatedAttention):
    def __init__(self, batch_size, dim, device):
        self.model = FlashAttention(causal=False, dropout=0.0, flash=True)
        self.model.to(device)
        self.batch_size = batch_size
        self.device = device


# inti testing
seq_lengths = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 64000]
batch_size = 32
dim = 512
device = "cuda:0"

dilated_tester = DilatedAttentionTest(batch_size, dim, device)
flash_tester = FlashAttentionTest(batch_size, dim, device)

dilated_times = []
flash_times = []

# test models on each sequence length
for seq_len in seq_lengths:
    dilated_time = dilated_tester.test(seq_len)
    dilated_times.append(dilated_time)

    flash_time = flash_tester.test(seq_len)
    flash_times.append(flash_time)

    print(
        f"Sequence lengths: {seq_len}, Dilated Attention time: {dilated_time}, flash Attention time: {flash_time}"
    )


# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(seq_lengths, dilated_times, marker="o", label="Dilated Attention")
plt.plot(seq_lengths, flash_times, marker="o", label="Flash Attention")
plt.title("Average forward pass time for different sequence lengths")
plt.xlabel("Sequence length")
plt.ylabel("Average forward pass time (seconds)")
plt.legend()
plt.grid(True)
plt.show()
