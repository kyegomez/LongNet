import torch
import time
from LongNet import DilatedLongNet

# Instantiate the DilatedLongNet model
model = DilatedLongNet()

# Define the input tensor
batch_size = 1
sequence_length = 512
input_tensor = torch.randn(batch_size, sequence_length).long()

# Enable CUDA if available
if torch.cuda.is_available():
    model = model.cuda()
    input_tensor = input_tensor.cuda()

# Measure the model forward pass speed
start_time = time.time()
_ = model(input_tensor)
end_time = time.time()

forward_pass_time = end_time - start_time
print(f"Model Forward Pass Time: {forward_pass_time} seconds")


# Count the number of parameters in the model
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of Model Parameters: {num_parameters}")
