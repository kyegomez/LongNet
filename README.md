[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)


# LongNet: Scaling Transformers to 1,000,000,000 Tokens
![LongNetBanner](longnet.jpg)


[![GitHub issues](https://img.shields.io/github/issues/kyegomez/LongNet)](https://github.com/kyegomez/LongNet/issues) 
[![GitHub forks](https://img.shields.io/github/forks/kyegomez/LongNet)](https://github.com/kyegomez/LongNet/network) 
[![GitHub stars](https://img.shields.io/github/stars/kyegomez/LongNet)](https://github.com/kyegomez/LongNet/stargazers) [![GitHub license](https://img.shields.io/github/license/kyegomez/LongNet)](https://github.com/kyegomez/LongNet/blob/master/LICENSE)
[![Share on Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Share%20%40kyegomez/LongNet)](https://twitter.com/intent/tweet?text=Excited%20to%20introduce%20LongNet,%20the%20all-new%20LongSequence%20model%20with%20the%20potential%20to%20revolutionize%20automation.%20Join%20us%20on%20this%20journey%20towards%20a%20smarter%20future.%20%23LongNet%20%23LongSequence&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FLongNet)
[![Share on Facebook](https://img.shields.io/badge/Share-%20facebook-blue)](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FLongNet)
[![Share on LinkedIn](https://img.shields.io/badge/Share-%20linkedin-blue)](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FLongNet&title=Introducing%20LongNet%2C%20the%20All-New%20LongSequence%20Model&summary=LongNet%20is%20the%20next-generation%20LongSequence%20model%20that%20promises%20to%20transform%20industries%20with%20its%20intelligence%20and%20efficiency.%20Join%20us%20to%20be%20a%20part%20of%20this%20revolutionary%20journey%20%23LongNet%20%23LongSequence&source=)
![Discord](https://img.shields.io/discord/999382051935506503)
[![Share on Reddit](https://img.shields.io/badge/-Share%20on%20Reddit-orange)](https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FLongNet&title=Exciting%20Times%20Ahead%20with%20LongNet%2C%20the%20All-New%20LongSequence%20Model%20%23LongNet%20%23LongSequence) [![Share on Hacker News](https://img.shields.io/badge/-Share%20on%20Hacker%20News-orange)](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FLongNet&t=Exciting%20Times%20Ahead%20with%20LongNet%2C%20the%20All-New%20LongSequence%20Model%20%23LongNet%20%23LongSequence)
[![Share on Pinterest](https://img.shields.io/badge/-Share%20on%20Pinterest-red)](https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FLongNet&media=https%3A%2F%2Fexample.com%2Fimage.jpg&description=LongNet%2C%20the%20Revolutionary%20LongSequence%20Model%20that%20will%20Change%20the%20Way%20We%20Work%20%23LongNet%20%23LongSequence)
[![Share on WhatsApp](https://img.shields.io/badge/-Share%20on%20WhatsApp-green)](https://api.whatsapp.com/send?text=I%20just%20discovered%20LongNet,%20the%20all-new%20LongSequence%20model%20that%20promises%20to%20revolutionize%20automation.%20Join%20me%20on%20this%20exciting%20journey%20towards%20a%20smarter%20future.%20%23LongNet%20%23LongSequence%0A%0Ahttps%3A%2F%2Fgithub.com%2Fkyegomez%2FLongNet)



This is an open source implementation for the paper [LongNet: Scaling Transformers to 1,000,000,000 Tokens](https://arxiv.org/abs/2307.02486) by Jiayu Ding, Shuming Ma, Li Dong, Xingxing Zhang, Shaohan Huang, Wenhui Wang, Furu Wei. The LongNet is a Transformer variant designed to scale sequence length up to more than 1 billion tokens without sacrificing performance on shorter sequences.


## Installation

```shell
pip install longnet
```

## Usage

Once you have installed LongNet, you can use the `DilatedAttention` class as follows:

```python
import torch
from long_net import DilatedAttention


# model config
dim = 512
heads = 8
dilation_rate = 2
segment_size = 64

# input data
batch_size = 32
seq_len = 8192


# create model and data
model = DilatedAttention(dim, heads, dilation_rate, segment_size, qk_norm=True)
x = torch.randn((batch_size, seq_len, dim))

output = model(x)
print(output)


```


# Train
- To run a simple training run on the enwiki8 dataset, gitclone, install the requirements.txt, and then run `python3 train.py`

## LongNet Summarized

Scaling sequence length has become a critical bottleneck in the era of large language models. However, existing methods struggle with either computational complexity or model expressivity, rendering the maximum sequence length restricted. In this paper, they introduce LongNet, a Transformer variant that can scale sequence length to more than 1 billion tokens, without sacrificing the performance on shorter sequences. Specifically, they propose dilated attention, which expands the attentive field exponentially as the distance grows.

## Features
LongNet has significant advantages:
1. It has a linear computation complexity and a logarithm dependency between tokens.
2. It can be served as a distributed trainer for extremely long sequences.
3. Its dilated attention is a drop-in replacement for standard attention, which can be seamlessly integrated with the existing Transformer-based optimization.

Experiment results demonstrate that LongNet yields strong performance on both long-sequence modeling and general language tasks. Their work opens up new possibilities for modeling very long sequences, e.g., treating a whole corpus or even the entire Internet as a sequence.


## Citation
```bibtex
@inproceedings{ding2023longnet,
  title={LongNet: Scaling Transformers to 1,000,000,000 Tokens},
  author={Ding, Jiayu and Ma, Shuming and Dong, Li and Zhang, Xingxing and Huang, Shaohan and Wang, Wenhui and Wei, Furu},
  booktitle={Proceedings of the 10th International Conference on Learning Representations},
  year={2023}
}
```

-----

# Todo

- [ ] Fix the ParallelTransformer Block's forward pass with dilated attn
- [ ] Train on enwiki 8 and test
- [ ] Create multihead iteration
