import torch 
from LongNet import LongNet

# Specify the device and dtype
device = "cuda:0"
dtype = torch.float16

# Specify the hyperparameters
d_model = 128
num_heads = 8
dilation_rates = [1, 2, 4, 8]
segment_sizes = [64, 64, 64, 64]

# Create the model
model = LongNet(
    d_model=d_model, 
    num_heads=num_heads, 
    dilation_rates=dilation_rates, 
    segment_sizes=segment_sizes
).to(device)

# Create some dummy data
x = torch.randn((64, 256, 128), device=device, dtype=dtype)

# Forward pass
output = model(x)

print(output.shape)  # Expected: [64, 256, 128]
