# Agora
This implementation of LongNet is brought to you by Agora, we're an all-new open source AI research organization with 1,500+ AI researchers all striving to advance Humanity!

![Agora banner](agora-banner-water.png)

[Join us and help contribute to LongNet and or recieve FAST support in the Agora discord!](https://discord.gg/qUtxnK2NMf)

# LongNet: Scaling Transformers to 1,000,000,000 Tokens

This is an open source implementation for the paper [LongNet: Scaling Transformers to 1,000,000,000 Tokens](https://arxiv.org/abs/2307.02486) by Jiayu Ding, Shuming Ma, Li Dong, Xingxing Zhang, Shaohan Huang, Wenhui Wang, Furu Wei. The LongNet is a Transformer variant designed to scale sequence length up to more than 1 billion tokens without sacrificing performance on shorter sequences.

## Introduction

Scaling sequence length has become a critical bottleneck in the era of large language models. However, existing methods struggle with either computational complexity or model expressivity, rendering the maximum sequence length restricted. In this paper, they introduce LongNet, a Transformer variant that can scale sequence length to more than 1 billion tokens, without sacrificing the performance on shorter sequences. Specifically, they propose dilated attention, which expands the attentive field exponentially as the distance grows.

## Features
LongNet has significant advantages:
1. It has a linear computation complexity and a logarithm dependency between tokens.
2. It can be served as a distributed trainer for extremely long sequences.
3. Its dilated attention is a drop-in replacement for standard attention, which can be seamlessly integrated with the existing Transformer-based optimization.

Experiment results demonstrate that LongNet yields strong performance on both long-sequence modeling and general language tasks. Their work opens up new possibilities for modeling very long sequences, e.g., treating a whole corpus or even the entire Internet as a sequence.

Here's the updated usage and installation section with two methods: git clone or pip install LongNet:

## Installation
c
You can install LongNet using one of the following methods:

### Method 1: Git Clone

1. Clone the LongNet repository from GitHub:

```shell
git clone https://github.com/kyegomez/LongNet.git
```

2. Navigate to the cloned directory:

```shell
cd LongNet
```

3. Prepare `flash_attn` library

```bash

cd flash_attn

python setup.py install

cd ..

```

4. Install the required dependencies:

```shell
pip install -r requirements.txt
```


### Method 2: Pip Install
* Note that pip install does not work as the `flash-attn` library cannot be compiled since it has custom CUDA Kernels and they need to be built manually.

1. Install LongNet directly from PyPI using pip:

```shell
pip install LongNet
```

Please note that LongNet requires a compatible Python version (tested with Python 3.7).

## Usage

Once you have installed LongNet, you can use the `DilatedAttention` class as follows:

```python
import torch
import torch.nn as nn
from LongNet import DilatedAttention

# Replace this with your correct GPU device
device = "cuda:0"
dtype = torch.float16

# Create an instance of DilatedAttention
d_model = 512
num_heads = 8
dilation_rate = 2
segment_size = 64
dropout = 0.2  # Specify the dropout rate
attention = DilatedAttention(
    d_model=d_model,
    num_heads=num_heads,
    dilation_rate=dilation_rate,
    segment_size=segment_size,
    dropout=dropout,
).to(device, dtype=dtype)

# Create some dummy input data
batch_size = 16
seq_len = 128
input_dim = d_model
inputs = torch.randn(batch_size, seq_len, input_dim, device=device, dtype=dtype)

# Forward pass
outputs = attention(inputs)

# Print the output shape
print(outputs.shape)  # Expected: [batch_size, seq_len, d_model]
```

# Training the Model
There are 2 methods, one is `accelerate` and the other `from LongNet import Train`

### Method 1 

* Git clone installation

* Init your parameters `accelerate config`

* Then `accelerate launch LongNet/training.py`

# Method 2

* Pip install method

```python

from LongNet import Train

Train()

```

In the example above, we create an instance of the `DilatedAttention` class with the specified hyperparameters. We then generate some dummy input data and pass it through the attention mechanism to obtain the outputs. Finally, we print the shape of the output tensor.


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
    def __init__(self, d_model, num_heads, dilation_rate, segment_size, dropout=0.0, causal=False, use_xpos=False, use_rel_pos_bias=False ):
        ...
```

## Parameters

- `d_model` (int): The dimensionality of the model. This should match the dimension of the input to the layer.

- `num_heads` (int): The number of attention heads to use in the `FlashMHA` attention mechanism.

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
dilated_attn = DilatedAttention(d_model=512, num_heads=8, dilation_rate=2, segment_size=64, dropout=0.1, causal=True, use_xpos=False, use_rel_pos_bias=False)
```

In this example, we're creating a `DilatedAttention` layer with a model dimensionality of 512, 8 attention heads, a dilation rate of 2, a segment size of 64, a dropout rate of 0.1, and causal masking enabled.

### Forward Pass

To perform a forward pass through the layer, simply call the instance as if it were a function, passing in your input tensor:

```python
import torch

# Assume x is your input tensor with shape (batch_size, sequence_length, d_model)
x = torch.rand(16, 1000, 512).to(device)

output = dilated_attn(x)
```

In this example, the input tensor `x` has a batch size of 16, a sequence length of 1000, and a model dimensionality of 512. The output tensor will have the same shape as the input tensor.

### Integration with Other Layers

You can integrate the `DilatedAttention` layer into a larger model just like any other PyTorch layer. For example, here's how you might use it as part of a simple transformer-like model:

```python
class SimpleTransformer(nn.Module):
    def __init__(self, d_model, num_heads, dilation_rate, segment_size, dropout):
        super().__init__()

        self.dilated_attn = DilatedAttention(d_model, num_heads, dilation_rate, segment_size, dropout, causal=True, use_xpos=False, use_rel_pos_bias=False)
        self.fc = nn.Linear(d_model, 10)  # Assume we're doing a 10-class classification task

    def forward(self, x):
        x = self.dilated_attn(x)
        x = self.fc(x[:, 0])  # Use the first position output as the "CLS" token
        return x

model = SimpleTransformer(d_model=512, num_heads=8, dilation_rate=2, segment_size=64, dropout=0.1)
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



## Share with Friends
Share LongNet with your friends and colleagues who might find it useful. Simply click on the links below to share on various platforms:

- [Facebook](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FLongNet)
- [Twitter](https://twitter.com/intent/tweet?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FLongNet&text=Check%20out%20the%20LongNet%20repository%2C%20an%20implementation%20for%20scaling%20Transformers%20to%201%2C000%2C000%2C000%20tokens.%20%23LongNet%20%23Transformers)
- [LinkedIn](https://www.linkedin.com/shareArticle?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FLongNet&title=LongNet%3A%20Scaling%20Transformers%20to%201%2C000%2C000%2C000%20Tokens)
- [Reddit](https://reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FLongNet&title=LongNet%3A%20Scaling%20Transformers%20to%201%2C000%2C000%2C000%20Tokens)
- [WhatsApp](https://wa.me/?text=Check%20out%20the%20LongNet%20repository%2C%20an%20implementation%20for%20scaling%20Transformers%20to%201%2C000%2C000%2C000%20tokens%3A%20https%3A%2F%2Fgithub.com%2Fkyegomez%2FLongNet)
- [Email](mailto:?subject=Check%20out%20the%20LongNet%20repository&body=Hey%2C%0A%0ACheck%20out%20the%20LongNet%20repository%2C%20an%20implementation%20for%20scaling%20Transformers%20to%201%2C000%2C000%2C000%20tokens%3A%0A%0Ahttps%3A%2F%2Fgithub.com%2Fkyegomez%2FLongNet%0A%0AEnjoy%21)
- [Hacker News](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FLongNet&t=LongNet%3A%20Scaling%20Transformers%20to%201%2C000%2C000%2C000%20Tokens)

Thank you for sharing!

[Share LongNet Repository](https://github.com/kyegomez/LongNet)




# Roadmap

* Test and evaluate and patch.

* And, create an interation of `DilatedAttention` with `FlashBlocksparseMHA`

* Create a multi-modal `DilationAttention` with multiway, sub layernorm, and xpos, sub layernorm, QK Layernorm, One write query head maybe

* Integrate Alibi and xpos for even further ridicoulus length extrapolation

* Recreate in Triton or Jax for ultra mega speed boost

* Integrate [Dynamic sparse flash attention](https://github.com/epfml/dynamic-sparse-flash-attention/blob/main/runtime-experiments/timeperf-hash-and-qk-sparse.ipynb) with DilatedAttention


## Citation
```
@inproceedings{ding2023longnet,
  title={LongNet: Scaling Transformers to 1,000,000,000 Tokens},
  author={Ding, Jiayu and Ma, Shuming and Dong, Li and Zhang, Xingxing and Huang, Shaohan and Wang, Wenhui and Wei, Furu},
  booktitle={Proceedings of the 10th International Conference on Learning Representations},
  year={2023}
}
```

