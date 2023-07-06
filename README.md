# LongNet: Scaling Transformers to 1,000,000,000 Tokens

This is an implementation for the paper [LongNet: Scaling Transformers to 1,000,000,000 Tokens](https://arxiv.org/abs/xxx.xxxxx) by Jiayu Ding, Shuming Ma, Li Dong, Xingxing Zhang, Shaohan Huang, Wenhui Wang, Furu Wei. The LongNet is a Transformer variant designed to scale sequence length up to more than 1 billion tokens without sacrificing performance on shorter sequences.

## Introduction

Scaling sequence length has become a critical demand in the era of large language models. However, existing methods struggle with either computational complexity or model expressivity, rendering the maximum sequence length restricted. In this paper, they introduce LongNet, a Transformer variant that can scale sequence length to more than 1 billion tokens, without sacrificing the performance on shorter sequences. Specifically, we propose dilated attention, which expands the attentive field exponentially as the distance grows.

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
from LongNet import LongNet
model = LongNet(config)
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
Feel free to share this repo with your friends and colleagues who might find it useful. Simply click on the link below:

[Share LongNet Repository](https://github.com/kyegomez/LongNet)
