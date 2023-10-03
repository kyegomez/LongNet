# import torch
# import torch.nn.functional as F
# from torch import nn
# from einops import rearrange

# class SelfAttention(nn.Module):
#     def __init__(self, dim):
#         super(SelfAttention, self).__init__()
#         # Linear transformations for Q, K, V
#         self.to_keys = nn.Linear(dim, dim, bias=False)
#         self.to_queries = nn.Linear(dim, dim, bias=False)
#         self.to_values = nn.Linear(dim, dim, bias=False)

#     def forward(self, x):
#         Q = self.to_queries(x)
#         K = self.to_keys(x)
#         V = self.to_values(x)
        
#         attn_weights = F.softmax(Q @ K.transpose(-2, -1), dim=-1)
#         attn_out = attn_weights @ V
#         return attn_out

# class DilatedAttention(nn.Module):
#     def __init__(self, dim, segment_length, dilated_rate):
#         super(DilatedAttention, self).__init__()
#         self.segment_length = segment_length
#         self.dilated_rate = dilated_rate
#         self.attention = SelfAttention(dim)
        
#     def forward(self, Q, K, V):
#         # Sparsify based on segment_length and dilated_rate
#         # This is a high-level representation. Actual implementation might be more complex.
#         Q = rearrange(Q, f'(b s) d -> b s d', s=self.segment_length)
#         K = rearrange(K, f'(b s) d -> b s d', s=self.segment_length)
#         V = rearrange(V, f'(b s) d -> b s d', s=self.segment_length)
        
#         # Apply self attention on sparsified segments
#         out = self.attention(Q, K, V)
#         out = rearrange(out, 'b s d -> (b s) d')
#         return out

# class MultiHeadDilatedAttention(nn.Module):
#     def __init__(self, dim, heads, segment_lengths, dilated_rates):
#         super(MultiHeadDilatedAttention, self).__init__()
        
#         num_heads = min(heads, len(segment_lengths), len(dilated_rates))
        
#         self.heads = nn.ModuleList([
#             DilatedAttention(dim, segment_lengths[i], dilated_rates[i])
#             for i in range(num_heads)
#         ])
        
#     def forward(self, Q, K, V):
#         out = [head(Q, K, V) for head in self.heads]
#         return torch.cat(out, dim=-1)

# class LONGNET(nn.Module):
#     def __init__(self, dim, heads, segment_lengths, dilated_rates):
#         super(LONGNET, self).__init__()
#         self.mha = MultiHeadDilatedAttention(dim, heads, segment_lengths, dilated_rates)
        
#     def forward(self, x):
#         # In actual scenario, we might need to project x to Q, K, V
#         return self.mha(x, x, x)

# # Placeholder variables
# DIM = 128
# HEADS = 4
# SEGMENT_LENGTHS = [4, 16, 8]
# DILATED_RATES = [1, 4, 2]

# model = LONGNET(DIM, HEADS, SEGMENT_LENGTHS, DILATED_RATES)
# x = torch.randn(32, 64, DIM)
# out = model(x)
# print(out)










# ###############

# import torch
# import torch.nn.functional as F
# from torch import nn
# from einops import rearrange

# class DilatedAttention(nn.Module):
#     def __init__(self, dim, segment_length, dilated_rate):
#         super(DilatedAttention, self).__init__()
#         self.segment_length = segment_length
#         self.dilated_rate = dilated_rate
#         self.to_keys = nn.Linear(dim, dim, bias=False)
#         self.to_queries = nn.Linear(dim, dim, bias=False)
#         self.to_values = nn.Linear(dim, dim, bias=False)
        
#     def forward(self, Q, K, V):
#         Q = self.to_queries(Q)
#         K = self.to_keys(K)
#         V = self.to_values(V)
        
#         # Segment and sparsify based on segment_length and dilated_rate
#         # This is a high-level representation. Actual implementation might be more complex.
#         Q = rearrange(Q, f'(b s) d -> b s d', s=self.segment_length)
#         K = rearrange(K, f'(b s) d -> b s d', s=self.segment_length)
#         V = rearrange(V, f'(b s) d -> b s d', s=self.segment_length)
        
#         # Apply self attention on sparsified segments
#         attn_weights = F.softmax(Q @ K.transpose(-2, -1), dim=-1)
#         attn_out = attn_weights @ V
#         attn_out = rearrange(attn_out, 'b s d -> (b s) d')
        
#         return attn_out

# # Define input dimensions
# BATCH_SIZE = 8
# SEQUENCE_LENGTH = 32
# DIM = 64
# SEGMENT_LENGTH = 4
# DILATED_RATE = 2

# # Create random input tensors
# Q = torch.randn(BATCH_SIZE * SEQUENCE_LENGTH, DIM)
# K = torch.randn(BATCH_SIZE * SEQUENCE_LENGTH, DIM)
# V = torch.randn(BATCH_SIZE * SEQUENCE_LENGTH, DIM)

# # Create the DilatedAttention model
# model = DilatedAttention(DIM, SEGMENT_LENGTH, DILATED_RATE)

# # Forward pass
# output = model(Q, K, V)
# print(output.shape)  # Should be [BATCH_SIZE * SEQUENCE_LENGTH, DIM]



############
# import torch
# import torch.nn.functional as F
# from torch import nn

# class DilatedAttention(nn.Module):
#     def __init__(self, dim, segment_length, dilated_rate):
#         super(DilatedAttention, self).__init__()
#         self.segment_length = segment_length
#         self.dilated_rate = dilated_rate
#         self.to_keys = nn.Linear(dim, dim, bias=False)
#         self.to_queries = nn.Linear(dim, dim, bias=False)
#         self.to_values = nn.Linear(dim, dim, bias=False)

#     def sparsify_segment(self, segment, dilated_rate):
#         return segment[::dilated_rate]

#     def forward(self, Q, K, V):
#         Q = self.to_queries(Q)
#         K = self.to_keys(K)
#         V = self.to_values(V)
        
#         b, n, _ = Q.size()
#         w = self.segment_length
#         r = self.dilated_rate
        
#         Qs, Ks, Vs = [], [], []
#         for i in range(0, n, w):
#             Q_segment = Q[:, i:i+w]
#             K_segment = K[:, i:i+w]
#             V_segment = V[:, i:i+w]
            
#             Qs.append(self.sparsify_segment(Q_segment, r))
#             Ks.append(self.sparsify_segment(K_segment, r))
#             Vs.append(self.sparsify_segment(V_segment, r))
        
#         Q_sparse = torch.cat(Qs, dim=1)
#         K_sparse = torch.cat(Ks, dim=1)
#         V_sparse = torch.cat(Vs, dim=1)
        
#         attn_weights = F.softmax(Q_sparse @ K_sparse.transpose(-2, -1), dim=-1)
#         attn_out = attn_weights @ V_sparse
        
#         return attn_out

# # Define input dimensions
# BATCH_SIZE = 8
# SEQUENCE_LENGTH = 32
# DIM = 64
# SEGMENT_LENGTH = 4
# DILATED_RATE = 2

# # Create random input tensors
# Q = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, DIM)
# K = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, DIM)
# V = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, DIM)

# # Create the DilatedAttention model
# model = DilatedAttention(DIM, SEGMENT_LENGTH, DILATED_RATE)

# # Forward pass
# output = model(Q, K, V)
# print(output.shape)  # Should be roughly [BATCH_SIZE, SEQUENCE_LENGTH/DILATED_RATE, DIM], but might vary depending on exact alignment



################
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

class DilatedAttention(nn.Module):
    def __init__(self, dim, segment_length, dilated_rate, dropout=0.1, qk_norm=True):
        super(DilatedAttention, self).__init__()
        self.segment_length = segment_length
        self.dilated_rate = dilated_rate
        self.qk_norm = qk_norm

        self.dropout = nn.Dropout(dropout)

        self.to_keys = nn.Linear(dim, dim, bias=False)
        self.to_queries = nn.Linear(dim, dim, bias=False)
        self.to_values = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)


        #softmax denominator
        self.softmax = nn.Softmax(dim=-1)

    def _segment_and_sparsify(self, x):
        # Divide x into segments
        x = rearrange(x, 'b (l s) d -> b l s d', s=self.segment_length)
        
        # Sparsify
        x = x[:, :, ::self.dilated_rate]
        
        # Flatten the segmented and sparsified tensor
        x = rearrange(x, 'b l s d -> b (l s) d')
        
        return x

    def forward(self, Q, K, V):
        Q, K, V = map(self._segment_and_sparsify, (Q, K, V))

        Q, K, V = self.to_queries(Q), self.to_keys(K), self.to_values(V)

        # norm qk values
        if self.qk_norm:
            Q, K = map(self.norm, (Q, K))
        else:
            pass

        # # Apply self attention on sparsified segments, use F.scaled_dot product in future
        # attn_weights = F.softmax(Q @ K.transpose(-2, -1), dim=-1)
        # attn_out = attn_weights @ V
        
        # return attn_out

        #flash
        with torch.backends.cuda.sdp_kernel(enable_math=True):
            out = F.scaled_dot_product_attention(Q, K, V)

            #dropout = 
            out = self.dropout(out)

            #softmax
            out = self.softmax(out)
        return out

# Define input dimensions
BATCH_SIZE = 8
SEQUENCE_LENGTH = 32
DIM = 64
SEGMENT_LENGTH = 4
DILATED_RATE = 2

# Create random input tensors
Q = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, DIM)
K = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, DIM)
V = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, DIM)

# Create the DilatedAttention model
model = DilatedAttention(DIM, SEGMENT_LENGTH, DILATED_RATE)

# Forward pass
output = model(Q, K, V)
print(output)
