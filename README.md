# Agora
This implementation of LongNet is brought to you by Agora, we're an all-new open source AI research organization with 1,500+ AI researchers all striving to advance Humanity!

![Agora banner](agora-banner-water.png)

[Join us and help contribute to LongNet and or recieve support in the Agora discord!](https://discord.gg/qUtxnK2NMf)

# LongNet: Scaling Transformers to 1,000,000,000 Tokens

This is an implementation for the paper [LongNet: Scaling Transformers to 1,000,000,000 Tokens](https://arxiv.org/abs/xxx.xxxxx) by Jiayu Ding, Shuming Ma, Li Dong, Xingxing Zhang, Shaohan Huang, Wenhui Wang, Furu Wei. The LongNet is a Transformer variant designed to scale sequence length up to more than 1 billion tokens without sacrificing performance on shorter sequences.

## Introduction

Scaling sequence length has become a critical bottleneck in the era of large language models. However, existing methods struggle with either computational complexity or model expressivity, rendering the maximum sequence length restricted. In this paper, they introduce LongNet, a Transformer variant that can scale sequence length to more than 1 billion tokens, without sacrificing the performance on shorter sequences. Specifically, we propose dilated attention, which expands the attentive field exponentially as the distance grows.

## Features
LongNet has significant advantages:
1. It has a linear computation complexity and a logarithm dependency between tokens.
2. It can be served as a distributed trainer for extremely long sequences.
3. Its dilated attention is a drop-in replacement for standard attention, which can be seamlessly integrated with the existing Transformer-based optimization.

Experiment results demonstrate that LongNet yields strong performance on both long-sequence modeling and general language tasks. Their work opens up new possibilities for modeling very long sequences, e.g., treating a whole corpus or even the entire Internet as a sequence.

Here's the updated usage and installation section with two methods: git clone or pip install LongNet:

## Installation

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

3. Install the required dependencies:

```shell
pip install -r requirements.txt
```

### Method 2: Pip Install

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

In the example above, we create an instance of the `DilatedAttention` class with the specified hyperparameters. We then generate some dummy input data and pass it through the attention mechanism to obtain the outputs. Finally, we print the shape of the output tensor.

## DilatedAttention Documentation

The `DilatedAttention` class implements dilated attention, which expands the attentive field exponentially as the distance between tokens grows. It inherits from `torch.nn.Module` and can be used as a drop-in replacement for standard attention mechanisms in Transformer models.

### Parameters

- `d_model` (int): The dimensionality of the input and output embeddings.
- `num_heads` (int): The number of attention heads.
- `dilation_rate` (int): The dilation rate for sparsifying the input sequence.
- `segment_size` (int): The size of each segment after sparsification.
- `dropout` (float, optional): The dropout probability to apply to the attention output. Default: 0.0 (no dropout).

### Inputs

- `x` (Tensor): The input tensor of shape `(batch_size, seq_len, d_model)`.

### Outputs

- `output` (Tensor): The output tensor of shape `(batch_size, seq_len, d_model)`.

Please note that the input tensor should be on the correct device (e.g., GPU) and have the appropriate data type (`dtype`).

## Citation
```
@inproceedings{ding2023longnet,
  title={LongNet: Scaling Transformers to 1,000,000,000 Tokens},
  author={Ding, Jiayu and Ma, Shuming and Dong, Li and Zhang, Xingxing and Huang, Shaohan and Wang, Wenhui and Wei, Furu},
  booktitle={Proceedings of the 10th International Conference on Learning Representations},
  year={2023}
}
```

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

* Integrate Alibi and xpos for even further ridicoulus length extrapolation

* Create a multi-modality verison with sub layer norm

* Integrate QK Layernorm

* Integrate One write query head maybe

* Recreate in Triton or Jax for ultra mega speed boost