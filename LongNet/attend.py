import math
from collections import namedtuple
from functools import wraps
from packaging import version

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

# constants

EfficientAttentionConfig = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# helpers

def exists(val):
    return val is not None

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

# main class

class FlashAttention(nn.Module):
    def __init__(
        self,
        causal = True,
        dropout = 0.,
        flash = True
    ):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.causal = causal
        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = EfficientAttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = EfficientAttentionConfig(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = EfficientAttentionConfig(False, True, True)

    def get_mask(self, i, j, device):
        return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)

    def flash_attn(self, q, k, v, mask = None, attn_bias = None):
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        # single headed key / values

        if k.ndim == 3:
            k = rearrange(k, 'b n d -> b 1 n d')

        if v.ndim == 3:
            v = rearrange(v, 'b n d -> b 1 n d')

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        if exists(mask) and mask.ndim != 4:
            mask = rearrange(mask, 'b j -> b 1 1 j')
            mask = mask.expand(-1, heads, q_len, -1)

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        causal = self.causal

        # handle attention bias

        if exists(attn_bias):
            mask_value = -torch.finfo(q.dtype).max // 2
            causal_mask = self.get_mask(q_len, k_len, device)
            attn_bias = attn_bias.masked_fill(causal_mask, mask_value)

            if exists(mask):
                attn_bias = attn_bias.masked_fill(~mask, mask_value)

            mask = attn_bias
            causal = False

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0., 
                is_causal = causal
            )

        return out

    def forward(self, q, k, v, mask = None, attn_bias = None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        scale = q.shape[-1] ** -0.5

        kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

        if self.flash:
            return self.flash_attn(q, k, v, mask = mask, attn_bias = attn_bias)

        # similarity

        sim = einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

        # attention bias

        if exists(attn_bias):
            sim = sim + attn_bias

        # causal mask

        if self.causal:
            causal_mask = self.get_mask(q_len, k_len, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)

        return out

class FlashMHA(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        factory_kwargs = {'device': device, 'dtype': dtype}

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True, **factory_kwargs)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True, **factory_kwargs)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True, **factory_kwargs)
        
        self.dropout_module = nn.Dropout(dropout).to(device=device, dtype=dtype)

        # Init flash attention
        self.flash_attention = FlashAttention(dropout=dropout, heads=num_heads, dropout=dropout, device=device, dtype=dtype)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)
    
    def forward(
            self,
            query,
            key,
            value,
            mask=None,
            attn_mask=None,
            incremental_state=None,
            is_first_step=False
        ):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Dict[str, Tensor]], Optional[bool]) -> Tuple[Tensor, Optional[Tensor]]
        tgt_len, bsz, embed_dim = query.size()

        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q *= self.scaling

        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, tgt_len, -1)
            mask = mask.view(mask.size(0) * self.num_heads, tgt_len, -1)
        
        attn_weights, _ = self.flash_attention(q, k, v, mask)
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, k.size(1)]

        attn_weights = self.dropout_module(attn_weights)
        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        return attn, attn_weights
