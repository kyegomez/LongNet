import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn


class StableAdamWUnfused(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=0.002,
        weight_decay=0.2,
        betas=(0.9, 0.99),
        eps=1e-8,
        clip_thresh=1.0,
        precision="amp_bfloat16",
        custom_scalar=65536,
    ):
        beta1, beta2 = betas[0], betas[1]
        defaults = dict(
            lr=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2
        )
        super(StableAdamWUnfused, self).__init__(params, defaults)

        self.eps = eps
        self.d = clip_thresh

        # Set precision to "custom_fp16" if you want to use a fixed loss scalar, custom_scalar, which is divided out in the update step.
        # If you do this, call (custom_scalar * loss).backward() instead of loss.backward().
        self.precision = precision
        self.custom_scaler = custom_scalar

        for group in self.param_groups:
            group["step"] = 1.0

        print("Using StableAdamWUnfused-v1")

    def __setstate__(self, state):
        super(StableAdamWUnfused, self).__setstate__(state)

    def step(self, closure=None):
        if closure is not None:
            closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            step = group["step"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                theta = p.data
                param_state = self.state[p]

                if self.precision == "custom_fp16":
                    g = p.grad.data / self.custom_scaler
                    if torch.any(torch.isnan(g) | torch.isinf(g)):
                        continue
                else:
                    g = p.grad.data

                if "exp_avg" not in param_state:
                    v = param_state["exp_avg"] = torch.zeros_like(theta)
                    u = param_state["exp_avg_sq"] = torch.zeros_like(theta)
                else:
                    v = param_state["exp_avg"]
                    u = param_state["exp_avg_sq"]

                beta1hat = (
                    beta1 * (1 - beta1 ** (step - 1)) / (1 - beta1**step)
                )
                beta2hat = (
                    beta2 * (1 - beta2 ** (step - 1)) / (1 - beta2**step)
                )

                v = v.mul_(beta1hat).add_(g, alpha=1.0 - beta1hat)
                u = u.mul_(beta2hat).addcmul_(g, g, value=1.0 - beta2hat)

                denominator = u.sqrt().add_(self.eps)

                # StableAdamW = AdamW + update clipping (https://arxiv.org/abs/1804.04235) applied tensor-wise.
                rms = (
                    torch.div(
                        g.pow(2),
                        torch.maximum(u, (self.eps**2) * torch.ones_like(u)),
                    )
                    .mean()
                    .sqrt()
                    .item()
                )

                theta = theta.mul_(1.0 - lr * weight_decay).addcdiv_(
                    v, denominator, value=-lr * (1.0 / max(1.0, rms / self.d))
                )

                # save current params
                param_state["exp_avg"] = v
                param_state["exp_avg_sq"] = u

            group["step"] = step + 1


class RelativePositionBias(nn.Module):
    def __init__(
        self, bidirectional=True, num_buckets=32, max_distance=128, n_heads=12
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.n_heads = n_heads
        self.relative_attention_bias = nn.Embedding(
            self.num_buckets, self.n_heads
        )

    @staticmethod
    def _relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, qlen, klen, step=None):
        step = 0 if step is None else step
        context_position = torch.arange(
            step,
            step + qlen,
            dtype=torch.long,
            device=self.relative_attention_bias.weight.device,
        )[:, None]
        memory_position = torch.arange(
            klen,
            dtype=torch.long,
            device=self.relative_attention_bias.weight.device,
        )[None, :]
        relative_position = (
            memory_position - context_position
        )  # shape (qlen, klen)

        rp_bucket = self._relative_position_bucket(
            relative_position,  # shape (qlen, klen)
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(
            rp_bucket
        )  # shape (qlen, klen, heads)
        values = values.permute([2, 0, 1]).unsqueeze(
            0
        )  # shape (1, heads, qlen, klen)
        return values

    def forward(self, batch_size, qlen, klen, step=None):
        # shape (batch * heads, qlen, klen)
        return (
            self.compute_bias(qlen, klen, step)
            .repeat(batch_size, 1, 1, 1)
            .view(-1, qlen, klen)
        )


def fixed_pos_embedding(x):
    seq_len, dim = x.shape
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim) / dim))
    sinusoid_inp = torch.einsum(
        "i , j -> i j", torch.arange(0, seq_len, dtype=torch.float), inv_freq
    ).to(x)
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(
        -2
    )  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\


def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m


def apply_rotary_pos_emb(x, sin, cos, scale=1):
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


class XPOS(nn.Module):
    def __init__(self, head_dim, scale_base=512):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.register_buffer(
            "scale",
            (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim),
        )

    def forward(self, x, offset=0, downscale=False):
        length = x.shape[1]
        min_pos = -(length + offset) // 2
        max_pos = length + offset + min_pos
        scale = (
            self.scale
            ** torch.arange(min_pos, max_pos, 1)
            .to(self.scale)
            .div(self.scale_base)[:, None]
        )
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]

        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, sin, cos, scale)
        return x


def SparsifyIndices(
    x: torch.Tensor, ws: List[int], rs: List[int], head_idx: int
) -> Tuple[int, torch.Tensor, Optional[torch.Tensor]]:
    b, n, c = x.size()

    print(f"x.size 1st: {x.shape} and xdtype: {x.dtype}")

    x_indices = torch.arange(0, n, dtype=torch.long, device=x.device)[
        None, :, None
    ]
    print(f"X indices dtype: {x_indices.shape} and dtype: {x.dtype}")

    num_subatt = sum([int(math.ceil(n / w)) for w in ws])
    max_subatt_n = min(n, max([w // r for w, r in zip(ws, rs)]))

    sparse_indices = -1 * torch.ones(
        (b, num_subatt * max_subatt_n, c), device=x.device, dtype=torch.int64
    )
    print(
        f"Sparse indices shape and dtype: {sparse_indices.shape} and dtype:"
        f" {sparse_indices.dtype}"
    )

    subatt_idx = 0
    for w, r in zip(ws, rs):
        for segment_indices in torch.split(x_indices, w, 1):
            offset = head_idx % r
            cur_sparse_indices = segment_indices[:, offset::r, :]
            print(
                f"Current sparse indices shape {cur_sparse_indices.shape} and"
                f" dtype: {cur_sparse_indices.dtype}"
            )
            start_idx = subatt_idx * max_subatt_n
            end_idx = start_idx + cur_sparse_indices.shape[1]
            sparse_indices[:, start_idx:end_idx] = cur_sparse_indices
            subatt_idx += 1

    if -1 in sparse_indices:
        padding_mask = sparse_indices[:, :, 0] != -1

        # to allow gather work for batching
        sparse_indices[~padding_mask] = 0

        # combine batch and subattention dims
        print(
            f"Padding mask shape: {padding_mask.shape} and dtype:"
            f" {padding_mask.dtype}"
        )
        padding_mask = padding_mask.view((-1, max_subatt_n))
    else:
        padding_mask = None

    return max_subatt_n, sparse_indices, padding_mask


def MixOutputs(
    out_shape: Tuple[int, int, int],
    out_dtype: torch.dtype,
    out_device: Union[torch.device, str],
    a_os: torch.Tensor,
    a_denoms: torch.Tensor,
    a_indices: torch.Tensor,
) -> torch.Tensor:
    print(f"Input 'a_os' shape: {a_os.shape} and dtype: {a_os.dtype}")
    print(
        f"Input 'a_denoms' shape: {a_denoms.shape} and dtype: {a_denoms.dtype}"
    )
    print(
        f"Input 'a_indices' shape: {a_indices.shape} and dtype:"
        f" {a_indices.dtype}"
    )

    # Ensure the source tensor has the same dtype as the target tensor before the scatter operation
    a_denoms = a_denoms.to(out_dtype)
    print(f"Converted 'a_denoms' dtype: {a_denoms.dtype}")

    # explicitly define the shape of att_denom_sums
    att_denom_sums_shape = (out_shape[0], out_shape[1])
    print(f"Att_denom_sums shape to be initialized: {att_denom_sums_shape}")

    # calculate sums of softmax denominators
    att_denom_sums = torch.zeros(
        att_denom_sums_shape, device=out_device, dtype=out_dtype
    )
    print(
        f"Initialized 'att_denom_sums' shape: {att_denom_sums.shape} and dtype:"
        f" {att_denom_sums.dtype}"
    )

    # Use scatter_add_ without unsqueezing a_denoms
    att_denom_sums.scatter_add_(1, a_indices[:, :, 0].squeeze(-1), a_denoms)

    # select attention softmax denominator sums for current sparse indices
    sparse_att_denom_sum = torch.gather(att_denom_sums, 1, a_indices[:, :, 0])
    print(
        f"'sparse_att_denom_sum' shape: {sparse_att_denom_sum.shape} and dtype:"
        f" {sparse_att_denom_sum.dtype}"
    )

    # compute alphas
    alphas = torch.divide(a_denoms, sparse_att_denom_sum)[:, :, None]
    print(f"Alphas shape: {alphas.shape} and dtype: {alphas.dtype}")

    out = torch.zeros(out_shape, dtype=out_dtype, device=out_device)
    print(f"Initialized 'out' shape: {out.shape} and dtype: {out.dtype}")

    out.scatter_add_(
        1,
        a_indices[:, :, : out.shape[2]],
        torch.multiply(a_os, alphas),
    )

    return out
