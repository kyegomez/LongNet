
import torch
import torch.nn as nn
import torch.nn.functional as F

from longnet.attend import FlashAttention
from longnet.utils import XPOS, MixOutputs, RelativePositionBias, SparsifyIndices


import math
from typing import List, Optional, Tuple, Union

device = "cuda:0"
dtype=torch.float16





def SparsifyIndices(
    x: torch.Tensor, ws: List[int], rs: List[int], head_idx: int
) -> Tuple[int, torch.Tensor, Optional[torch.Tensor]]:
    b, n, c = x.size()

    print(f'x.size 1st: {x.shape} and xdtype: {x.dtype}')

    x_indices = torch.arange(0, n, dtype=torch.long, device=x.device)[None, :, None]
    print(f"X indices dtype: {x_indices.shape} and dtype: {x.dtype}")

    num_subatt = sum([int(math.ceil(n / w)) for w in ws])
    max_subatt_n = min(n, max([w // r for w, r in zip(ws, rs)]))

    sparse_indices = -1*torch.ones((b, num_subatt * max_subatt_n, c), device=x.device, dtype=torch.int64)
    print(f"Sparse indices shape and dtype: {sparse_indices.shape} and dtype: {sparse_indices.dtype}")

    subatt_idx = 0
    for w, r in zip(ws, rs):
        for segment_indices in torch.split(x_indices, w, 1):
            offset = head_idx % r
            cur_sparse_indices = segment_indices[:, offset::r, :]
            print(f"Current sparse indices shape {cur_sparse_indices.shape} and dtype: {cur_sparse_indices.dtype}")
            start_idx = subatt_idx*max_subatt_n
            end_idx = start_idx+cur_sparse_indices.shape[1]
            sparse_indices[:, start_idx:end_idx] = cur_sparse_indices
            subatt_idx += 1

    if -1 in sparse_indices:
        padding_mask = sparse_indices[:, :, 0] != -1

        # to allow gather work for batching
        sparse_indices[~padding_mask] = 0

        # combine batch and subattention dims
        print(f"Padding mask shape: {padding_mask.shape} and dtype: {padding_mask.dtype}")
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
    print(f"Input 'a_denoms' shape: {a_denoms.shape} and dtype: {a_denoms.dtype}")
    print(f"Input 'a_indices' shape: {a_indices.shape} and dtype: {a_indices.dtype}")
    
    # Ensure the source tensor has the same dtype as the target tensor before the scatter operation
    a_denoms = a_denoms.to(out_dtype)
    print(f"Converted 'a_denoms' dtype: {a_denoms.dtype}")

    # explicitly define the shape of att_denom_sums
    att_denom_sums_shape = (out_shape[0], out_shape[1])
    print(f"Att_denom_sums shape to be initialized: {att_denom_sums_shape}")
    
    # calculate sums of softmax denominators
    att_denom_sums = torch.zeros(att_denom_sums_shape, device=out_device, dtype=out_dtype)
    print(f"Initialized 'att_denom_sums' shape: {att_denom_sums.shape} and dtype: {att_denom_sums.dtype}")
    
    # a_indices = a_indices[:, :, 0].squeeze(-1).squeeze(-1)
    att_denom_sums.scatter_add_(1, a_indices[:, :, 0].squeeze(-1), a_denoms)

    # select attention softmax denominator sums for current sparse indices
    sparse_att_denom_sum = torch.gather(att_denom_sums, 1, a_indices[:, :, 0].squeeze(-1))
    print(f"'sparse_att_denom_sum' shape: {sparse_att_denom_sum.shape} and dtype: {sparse_att_denom_sum.dtype}")

    # compute alphas
    alphas = torch.divide(a_denoms, sparse_att_denom_sum)[:, :, None]
    print(f"Alphas shape: {alphas.shape} and dtype: {alphas.dtype}")

    out = torch.zeros(out_shape, dtype=out_dtype, device=out_device)
    print(f"Initialized 'out' shape: {out.shape} and dtype: {out.dtype}")

    out.scatter_add_(
        1,
        a_indices[:, :, :out.shape[2]],
        torch.multiply(a_os, alphas),
    )

    return out




class ParallelWrapper:
    """
    A simple wrapper to enable easy usage of data parallelism.

    Arguments:
        model: The neural network model to be parallelized.
        device (optional): The device to which the model should be moved. Default: "cuda".
        use_data_parallel (optional): A boolean flag to indicate whether to use data parallelism or not. Default: True.
    """
    def __init__(
            self,
            model,
            device="cuda",
            use_data_parallel=True
    ):
        self.model = model.to(device)
        self.use_data_parallel = use_data_parallel
        self.device = device

        if self.use_data_parallel and torch.cuda.device_count() < 1:
            print(f"Using {torch.cuda.device_count()} GPUS")
            self.model = nn.DataParallel(self.model)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def to(self, device):
        self.device = device
        self.model= self.model.to(device)
        return self
    
    def __getattr__(self, name):
        #redirect attribute access to the internal model to allow direct access to its methods and props
        return getattr(self.model, name)
    



#add alibi, qk layer norm, one write head, multiway, 
class DilatedAttentionNew(nn.Module):
    """
    Dilated Attention Module.

    Arguments:
        d_model: The dimension of the attention layers.
        num_heads: The number of attention heads.
        dilation_rate: The dilation rate for dilated attention.
        segment_size: The segment size for dilated attention.
        dropout (optional): The dropout probability. Default: 0.0
        casual (optional): If set to True, the attention mechanism is casual. Default: False
        use_xpos (optional): If set to True, xpos is used for positional encoding. Default: False
        use_rel_pos_bias (optional): If set to True, relative position bias is used in the attention mechanism. Default: False

    Usage:
        The `DilatedAttention` class can be used as a module for neural networks and is especially suited for transformer architectures.

        Example:
            attention = DilatedAttention(d_model=512, num_heads=8, dilation_rate=2, segment_size=64, use_xpos=True, use_rel_pos_bias=True)
            output = attention(input_tensor)

        This will return the output tensor after applying dilated attention. The `use_xpos` and `use_rel_pos_bias` parameters allow for switching on positional encoding and relative positional bias respectively.
    """
    def __init__(self, d_model, num_heads, dilation_rate, segment_size, dropout=0.0, casual=False, use_xpos=False, use_rel_pos_bias=False):
        super(DilatedAttentionNew, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.dilation_rate = dilation_rate
        self.segment_size = segment_size

        self.dropout = nn.Dropout(dropout)
        self.casual = casual

        self.use_xpos = use_xpos
        self.use_rel_pos_bias = use_rel_pos_bias

        self.attention = FlashAttention(causal=self.casual, dropout=dropout).to(device)

        if use_xpos:
            self.xpos = XPOS(head_dim=d_model//num_heads)
        if use_rel_pos_bias:
            self.relative_bias = RelativePositionBias(num_buckets=32, max_distance=128, n_heads=num_heads)

        #head offsets
        self.head_offsets = nn.Parameter(torch.randn(num_heads, d_model))

    def get_mask(self, i, j):
        return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 2)

    def forward(self, x):
        # get dimensions
        batch_size, seq_len, _ = x.shape
        print(f"X shape: {x.shape} and dtype: {x.dtype}")

        # calculate the necessary padding
        padding_len = -seq_len % self.segment_size
        x = F.pad(x, (0,0,0,padding_len))
        print(f"f x after pad: {x.shape} and dtype: {x.dtype}")
        seq_len = seq_len + padding_len

        if self.use_xpos:
            x = self.xpos(x)
            print(f"XPOS shape and dtype: {x.shape} and dtype: {x.dtype}")


        head_idx = int(self.head_offsets[0, 0].item())
        print(f"head_idx: {head_idx}")

        # Prepare sparse indices
        # max_subatt_n, sparse_indices, padding_mask = sparsify_indices(x, [self.segment_size], [self.dilation_rate], self.head_offsets)
        max_subatt_n, sparse_indices, padding_mask = SparsifyIndices(x, [self.segment_size], [self.dilation_rate], head_idx)

        # Split and sparsify
        x = x.view(batch_size, -1, self.segment_size, self.d_model)
        print(f"Split and sparsify x: {x.shape} and dtype: {x.dtype}")

        #Gather operation
        x_dim1 = x.size(1)
        x = x.gather(2, sparse_indices[:, :x_dim1, :].unsqueeze(-1).expand(-1, -1, -1, self.d_model))
        print(f"gather op: {x.shape} and xdtype: {x.dtype}")


        # Perform attention
        attn_output = self.attention(x, x, x)
        print(f"attn output shape and type: {attn_output.shape} and dtype: {attn_output.dtype}")

        #if use rel pos => apply relative positioning bias 
        if self.use_rel_pos_bias:
            attn_output += self.relative_bias(batch_size, attn_output.size(1), attn_output.size(1))
            print(f"attn_output shape and dtype: {attn_output.shape} and dtype: {attn_output.dtype}")

        # if casual create a mask and apply to the output
        if self.casual:
            mask = self.get_mask(attn_output.size(1), attn_output.size(1))
            attn_output = attn_output.masked_fill(mask, float('-inf'))

        # apply dropout
        attn_output = self.dropout(attn_output)

        # Mix outputs
        attn_output = MixOutputs((batch_size, seq_len, self.d_model), x.dtype, x.device, attn_output, attn_output.sum(dim=-1), sparse_indices)
        print(f"Attn output dtype and shape: {attn_output.shape}")

        return attn_output









#add alibi, qk layer norm, one write head, multihway, 
class DilatedAttention(nn.Module):
    """
    Dilated Attention Module.

    Arguments:
        d_model: The dimension of the attention layers.
        num_heads: The number of attention heads.
        dilation_rate: The dilation rate for dilated attention.
        segment_size: The segment size for dilated attention.
        dropout (optional): The dropout probability. Default: 0.0
        casual (optional): If set to True, the attention mechanism is casual. Default: False
        use_xpos (optional): If set to True, xpos is used for positional encoding. Default: False
        use_rel_pos_bias (optional): If set to True, relative position bias is used in the attention mechanism. Default: False

    Usage:
        The `DilatedAttention` class can be used as a module for neural networks and is especially suited for transformer architectures.

        Example:
            attention = DilatedAttention(d_model=512, num_heads=8, dilation_rate=2, segment_size=64, use_xpos=True, use_rel_pos_bias=True)
            output = attention(input_tensor)

        This will return the output tensor after applying dilated attention. The `use_xpos` and `use_rel_pos_bias` parameters allow for switching on positional encoding and relative positional bias respectively.
    """
    def __init__(self, d_model, num_heads, dilation_rate, segment_size, dropout=0.0, casual=False, use_xpos=False, use_rel_pos_bias=False):
        super(DilatedAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.dilation_rate = dilation_rate
        self.segment_size = segment_size

        self.dropout = nn.Dropout(dropout)
        self.casual = casual

        self.use_xpos = use_xpos
        self.use_rel_pos_bias = use_rel_pos_bias

        self.attention = FlashAttention(causal=self.casual, dropout=dropout).to(device)

        if use_xpos:
            self.xpos = XPOS(head_dim=d_model//num_heads)
        if use_rel_pos_bias:
            self.relative_bias = RelativePositionBias(num_buckets=32, max_distance=128, n_heads=num_heads)

        #head offsets
        self.head_offsets = nn.Parameter(torch.randn(num_heads, d_model))

    def get_mask(self, i, j):
        return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 2)

    def forward(self, x):
        print(f"X original shape: {x.shape} and x dtype: {x.dtype}")

        batch_size, seq_len, _ = x.shape
        padding_len = -seq_len % self.segment_size
        x = F.pad(x, (0,0,0,padding_len))
        seq_len = seq_len + padding_len
        print(f"Paddex x shape:  {x.shape}")
        

        if self.use_xpos:
            x = self.xpos(x)
        
        # Split and sparsify
        x = x.view(batch_size, -1, self.segment_size, self.d_model)
        print(f"z after view shape: {x.shape}")

        x = x[:, :, :: self.dilation_rate, :]
        print(f"x after dilation shape: {x.shape} and x.dtype: {x.dtype}")

        # Perform attention
        attn_output = self.attention(x, x, x)
        print(f"Attn output: {attn_output.shape} and dtype: {attn_output.dtype}")

        #if use rel pos => apply relative positioning bias 
        if self.use_rel_pos_bias:
            attn_output += self.relative_bias(batch_size, attn_output.size(1), attn_output.size(1))
            print(f"attn_output: {attn_output.shape} and attn output: {attn_output.dtype}")

        # if casual create a mask and apply to the output
        if self.casual:
            mask = self.get_mask(attn_output.size(1), attn_output.size(1))
            print(f"mask shape: {mask.shape} and mask dtype: {x.dtype}")

            attn_output = attn_output.masked_fill(mask, float('-inf'))
            print(f"attn output shape: {attn_output.shape} and attn_output: {attn_output.dtype}")

        # apply dropout
        attn_output = self.dropout(attn_output)
        print(f"attn output after dropout: {attn_output.shape} and dtype: {attn_output.dtype}")

        # Scatter and concatenate 
        attn_output = attn_output.reshape(batch_size, -1, self.d_model)
        print(f"attn_output scatter and concatenate: {attn_output.shape} and {attn_output.dtype}")
        return attn_output












class MultiHeadDilatedAttention(DilatedAttention):
    """
    **Description:**  
    The `MultiHeadDilatedAttention` class offers a multi-head version of dilated attention, allowing the model to focus on various parts of an input sequence with different patterns.

    **Parameters:**  
    - `d_model`: Dimension of the model. This is usually a multiple of `num_heads`.
    - `num_heads`: Number of attention heads.
    - `dilation_rate`: The rate of dilation for the dilated attention mechanism.
    - `segment_size`: The size of the segment for dilated attention.

    Additionally, the parameters from the parent class `DilatedAttention` are inherited.

    **Methods:**  
    - `sparsify(x, sj, head_layer)`: Transforms the tensor by applying the dilated attention offset based on the given head layer. It's used internally and typically not called directly.
    - `forward(x)`: Computes the multi-head dilated attention on the input tensor `x`.
    """
    def __init__(
            self,
            d_model,
            num_heads,
            dilation_rate,
            segment_size,
            *args,
            **kwargs,
    ):
        super(MultiHeadDilatedAttention, self).__init__(d_model,
                                                        num_heads,
                                                        dilation_rate,
                                                        segment_size, 
                                                        *args,
                                                        **kwargs)
        
        self.head_dim  = d_model // num_heads

        #define linear layers for Q, K, V projections for each head
        self.query_layers = nn.ModuleList([nn.Linear(d_model, self.head_dim) for _ in range(num_heads)])
        self.key_layers = nn.ModuleList([nn.Linear(d_model, self.head_dim) for _ in range(num_heads)])
        self.value_layers = nn.ModuleList([nn.Linear(d_model, self.head_dim) for _ in range(num_heads)])

        #define linear layer for concat the multi head outputs
        self.concat_layer = nn.Linear(num_heads * self.head_dim, d_model)

    def sparsify(self, x, sj, head_layer):

        x_proj = head_layer(x)
        batch_size, seq_len, _ = x_proj.shape

        padding_len = -seq_len % self.segment_size
        x_proj = F.pad(x_proj, (0, 0, 0, padding_len))
        seq_len += padding_len

        x_proj = x_proj.view(batch_size, -1, self.segment_size, self.head_dim)
        x_proj = x_proj[:, :, sj::self.dilation_rate, :]

        return x_proj
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        all_heads_output = []
        
        for j in range(self.num_heads):
            sj = j % self.dilation_rate

            #create Q, K, V for each head based on offset
            Qi = self.sparsify(x, sj, self.query_layers[j])
            Ki = self.sparsify(x, sj, self.key_layers[j])
            Vi = self.sparsify(x, sj, self.value_layers[j])


            #combine q, k, v tensors for base class forward method
            combined_input = torch.cat([Qi, Ki, Vi], dim=-1)

            #apply inherited atention on q, k, v
            head_output = super().forward(Qi, Ki, Vi)
            all_heads_output.append(head_output)

        #concat multi head outputs and pdd through final linear layer 
        concat_heads = torch.cat(all_heads_output, dim=-1)
        return self.concat_layer(concat_heads)



