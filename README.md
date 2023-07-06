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

## Installation
```
git clone https://github.com/kyegomez/LongNet.git
cd LongNet
pip install -r requirements.txt
```
Note: Please ensure you have a compatible Python version installed. This codebase has been tested with Python 3.7.

## Usage
```python
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
```

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
