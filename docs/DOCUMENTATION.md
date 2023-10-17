

# DilatedAttention Documentation

## Algorithmic Psueodocode:
```
1. Initialize the input (Q, K, V) and split them into segments {(Qei, Kei, Vei)} with equal segment length w.
2. Sparsify each segment along the sequence dimension by selecting the rows with an interval r.
3. Feed the sparsified segments into the attention in parallel.
4. Scatter and concatenate the output O from the attention.
5. Implement a mixture of dilated attentions with different segment sizes and dilation rates {ri, wi}.
6. For multi-head dilated attention, differ the computation among different heads by sparsifying different parts of the query-key-value pairs.
7. Concatenate the outputs of different heads into a final output.
```


## Class Definition

```python
class DilatedAttention(nn.Module):
    def __init__(self, dim, heads, dilation_rate, segment_size, dropout=0.0, causal=False, use_xpos=False, use_rel_pos_bias=False ):
        ...
```

## Parameters

- `dim` (int): The dimensionality of the model. This should match the dimension of the input to the layer.

- `heads` (int): The number of attention heads to use in the `FlashMHA` attention mechanism.

- `dilation_rate` (int): The dilation rate to use when processing the input sequence. Larger values will result in fewer, but wider, attention computations.

- `segment_size` (int): The size of the segments into which the input sequence is divided before dilating and computing attention.

- `dropout` (float, optional): The dropout rate to apply to the attention outputs. Default is 0.0.

- `causal` (bool, optional): If True, a causal mask will be applied to the attention outputs, preventing any given position from attending to future positions. Default is False.

- `use_xpos` (optional): If set to True, xpos is used for positional encoding. Default: False

- `use_rel_pos_bias` (optional): If set to True, relative position bias is used in the attention mechanism. Default: False

## Usage

### Creating an Instance

First, you need to create an instance of the `DilatedAttention` class. Here is how you do it:

```python
dilated_attn = DilatedAttention(dim=512, heads=8, dilation_rate=2, segment_size=64, dropout=0.1, causal=True, use_xpos=False, use_rel_pos_bias=False)
```

In this example, we're creating a `DilatedAttention` layer with a model dimensionality of 512, 8 attention heads, a dilation rate of 2, a segment size of 64, a dropout rate of 0.1, and causal masking enabled.

### Forward Pass

To perform a forward pass through the layer, simply call the instance as if it were a function, passing in your input tensor:

```python
import torch

# Assume x is your input tensor with shape (batch_size, sequence_length, dim)
x = torch.rand(16, 1000, 512).to(device)

output = dilated_attn(x)
```

In this example, the input tensor `x` has a batch size of 16, a sequence length of 1000, and a model dimensionality of 512. The output tensor will have the same shape as the input tensor.

### Integration with Other Layers

You can integrate the `DilatedAttention` layer into a larger model just like any other PyTorch layer. For example, here's how you might use it as part of a simple transformer-like model:

```python
class SimpleTransformer(nn.Module):
    def __init__(self, dim, heads, dilation_rate, segment_size, dropout):
        super().__init__()

        self.dilated_attn = DilatedAttention(dim, heads, dilation_rate, segment_size, dropout, causal=True, use_xpos=False, use_rel_pos_bias=False)
        self.fc = nn.Linear(dim, 10)  # Assume we're doing a 10-class classification task

    def forward(self, x):
        x = self.dilated_attn(x)
        x = self.fc(x[:, 0])  # Use the first position output as the "CLS" token
        return x

model = SimpleTransformer(dim=512, heads=8, dilation_rate=2, segment_size=64, dropout=0.1)
```

In this example, we first pass the input tensor through the `DilatedAttention` layer, then we pass the output of the first position through a fully-connected layer to perform a classification task.


## DilationAttention Overview

`DilatedAttention` is a neural network architecture that incorporates attention mechanisms, specifically the multi-head attention, in a dilated manner. The main idea behind this architecture is to leverage the efficient attention calculation capabilities of the `FlashMHA` method, which is part of the `flash_attn` module, while also providing the ability to handle longer sequences with reduced computation through dilation.

## Components

The class `DilatedAttention` has the following primary components:

- **FlashMHA attention**: A fast and efficient multi-head attention mechanism implemented using the `FlashMHA` method. This is the main attention computation method used in the architecture.

- **Dilation**: Dilating the input sequences allows the model to handle longer sequences with fewer computations, making the architecture more scalable and efficient.

- **Causal masking (optional)**: If the `causal` argument is set to `True`, a causal mask is applied to the attention outputs, ensuring that each output position only depends on earlier positions in the sequence. This feature is particularly useful when dealing with sequential data where future dependencies should not be considered.

- **Dropout**: A dropout layer that can be configured to add regularization to the model and prevent overfitting.

## How It Works

The `DilatedAttention` model works in the following steps:

1. **Input Reshape**: Reshapes the input into smaller segments based on the provided `segment_size` and then dilates it by selecting every `dilation_rate` segment.

2. **Attention Computation**: Uses `FlashMHA` to compute the attention over the dilated segments.

3. **Causal Masking**: If `causal` is set to `True`, a causal mask is applied to the attention output. This ensures that the output at each position in the sequence does not depend on any future positions.

4. **Dropout**: Applies dropout to the attention outputs as a means of regularization.

5. **Output Reshape**: Reshapes the output to match the original sequence length, concatenating the dilated segments.

## Why It Works

The `DilatedAttention` model achieves efficiency and scalability in several ways:

- **Efficient attention calculation**: The use of `FlashMHA` enables efficient and fast attention computation.

- **Dilation**: Dilation allows the model to handle longer sequences with reduced computation, effectively making the model more scalable.

- **Causal masking**: By ensuring that each output position only depends on earlier positions in the sequence, the model becomes suitable for tasks involving sequential data.

## Potential Optimizations

1. **Parallelization**: Take advantage of the parallel processing capabilities of modern GPUs for the dilation and reshaping steps.

2. **Memory optimization**: Efficient memory usage could be achieved through gradient checkpointing or activation pruning.

3. **Pre-computation**: If some portions of the input data remain constant across multiple operations, pre-compute those portions and store the results for reuse.

4. **Batch normalization**: Incorporating batch normalization layers could help to speed up the learning process and improve generalization.

5. **Pruning and Quantization**: Pruning unnecessary connections and quantizing the model parameters can help in reducing the model's memory footprint and speed up computation without sacrificing much accuracy.

In the example above, we create an instance of the `DilatedAttention` class with the specified hyperparameters. We then generate some dummy input data and pass it through the attention mechanism to obtain the outputs. Finally, we print the shape of the output tensor.



### DilatedAttention

DilatedAttention is a module that performs dilated attention on input tensors.

#### Systems Understanding

The DilatedAttention module takes an input tensor of shape (batch_size, sequence_length, dim) and applies dilated attention to the input. The attention mechanism consists of multiple attention heads with different dilation rates. Each attention head operates on a subset of the input tensor determined by the dilation rate and segment size. The outputs of the attention heads are weighted and combined to produce the final output tensor.

The input tensor is first passed through the positional encoding layer, which adds positional information to the input. Then, for each attention head, the input tensor is divided into segments based on the dilation rate. Each segment is fed into a FlashMHA (Flash Multi-Head Attention) module, which performs self-attention on the segment. The attention outputs from each segment are concatenated and form the output of the attention head. If relative positional bias is enabled, the relative positional bias is added to the attention outputs. If casual attention is enabled, a mask is applied to the attention outputs to prevent attending to future positions. The attention outputs are then multiplied by weights corresponding to the dilation rates and combined to form the final output tensor.

#### Usage example

```python
import torch
from LongNet import DilatedAttention

# Create a DilatedAttention module
dilated_attention = DilatedAttention(dim=512, heads=8, dilation_rate=2, segment_size=16)

# Generate random input tensor
batch_size = 4
sequence_length = 32
dim = 512
input_tensor = torch.randn(batch_size, sequence_length, dim)

# Apply dilated attention to the input tensor
output_tensor = dilated_attention(input_tensor)

# Print the shape of the output tensor
print(output_tensor.shape)
```

#### Constructor

```python
def __init__(
    self,
    dim: int,
    heads: int,
    dilation_rate: int,
    segment_size: int,
    dropout: float = 0.0,
    casual: bool = False,
    use_xpos: bool = False,
    use_rel_pos_bias: bool = False,
    Distributed: bool = False
)
```

Initialize the DilatedAttention module.

**Args:**

- `dim` (int): The dimension of the model.
- `heads` (int): The number of attention heads.
- `dilation_rate` (int): The dilation rate.
- `segment_size` (int): The segment size.
- `dropout` (float, optional): The dropout rate. Defaults to 0.0.
- `casual` (bool, optional): Whether to use casual attention. Defaults to False.
- `use_xpos` (bool, optional): Whether to use positional encoding. Defaults to False.
- `use_rel_pos_bias` (bool, optional): Whether to use relative positional bias. Defaults to False.
- `Distributed` (bool, optional): Whether to use distributed computation. Defaults to False.

#### Methods

##### forward

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Perform forward pass through the DilatedAttention module.

**Args:**

- `x` (torch.Tensor): The input tensor of shape (batch_size, sequence_length, dim).

**Returns:**

- `torch.Tensor`: The output tensor of shape (batch_size, sequence_length, dim).