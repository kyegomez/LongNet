import math

#distributed dilated attention based on second iteration
import torch 
import torch.nn as nn

import time
# Replace this with your correct GPU device
device = "cuda:0"
dtype=torch.float16





class DilatedAttentionLLAMA(nn.Module):
    def __init__(self, num_heads, head_dim, hidden_size, block_size=1024, dilations=[1,2,4,8]):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.block_size = block_size
        self.dilations = dilations
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, query_states, key_states, value_states, attention_mask=None):
        bsz, _, q_len = query_states.size()
        attn_output = torch.zeros_like(query_states)
        w = torch.zeros(bsz, self.num_heads, q_len, 1, device=query_states.device, dtype=torch.float32)
        des = []
        rs = []

        for d in self.dilations:
            qs = torch.cat(torch.split(query_states[:,:,::d], self.block_size, dim=2))
            ks = torch.cat(torch.split(key_states[:,:,::d], self.block_size, dim=2))
            vs = torch.cat(torch.split(value_states[:,:,::d], self.block_size, dim=2))

            attn_weights = torch.matmul(qs, ks.transpose(1, 2)) / math.sqrt(self.head_dim)  # Change here

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
                )

            de = torch.cat(attn_weights.detach().exp().sum(dim=-1, keepdim=True).split(bsz, dim=0), dim=2)
            des.append(de)

            w[:,:,::d] += de

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            a = torch.matmul(attn_weights, vs)
            a = torch.concat(a.split(bsz, dim=0), dim=2)
            rs.append(a)

        for d, de, a in zip(self.dilations, des, rs):
            aw = de / w[:,:,::d]
            s = aw * a

            idx = torch.arange(0, attn_output.shape[2], step=d, dtype=torch.int64, device=s.device).view(1, 1, s.shape[2], 1).expand_as(s)
            attn_output = attn_output.scatter_add(dim=2, index=idx, src=s.to(query_states.dtype))

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output


# Test
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bsz = 32
seq_len = 64
d_model = 512
num_heads = 8
head_dim = d_model // num_heads
hidden_size = d_model

x = torch.randn(bsz, seq_len, d_model).to(device)
model = DilatedAttentionLLAMA(num_heads, head_dim, hidden_size).to(device)

start_time = time.time()
output = model(x, x, x)
end_time = time.time()

print(f"Time taken for forward pass: {end_time - start_time} seconds")
print(f"Output shape: {output.shape}")
