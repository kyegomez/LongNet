import timeit
import torch 
from LongNet.attention import ParallelWrapper, DilatedAttention

#model condig
d_model = 512
num_heads = 8
dilation_rate = 2
segment_size = 64


device="cuda:0"
dtype=torch.float16

#inputs
batch_size = 32
seq_len = 8192


#create model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DilatedAttention(
    d_model,
    num_heads,
    dilation_rate,
    segment_size
)
parallel_model = ParallelWrapper(model, device=device)

x = torch.randn((batch_size, seq_len, d_model), device=device, dtype=dtype)

#test forward pass
with torch.no_grad():
    output = model(x)
    print(f"Output shape: {output.shape}") #expected (batch_size, seq_len)

#benchmark model
num_runs = 1000
start_time = timeit.default_timer()
for _ in range(num_runs):
    model(x)


elapsed_time = timeit.default_timer() - start_time
print(f"Average forward pass time: {elapsed_time / num_runs:.6f} seconds")