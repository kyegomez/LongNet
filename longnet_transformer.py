import torch
from long_net.model import LongNetTransformer

longnet = LongNetTransformer(
    num_tokens=20000,
    dim=512,
    depth=6,
    dim_head=64,
    heads=8,
    ff_mult=4,
)

tokens = torch.randint(0, 20000, (1, 512))
logits = longnet(tokens)
print(logits)
