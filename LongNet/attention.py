import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel

from LongNet.utils import XPOS, RelativePositionBias

from LongNet.attend import FlashAttention

device = "cuda:0"
dtype=torch.float16



# Define the attention module
class DilatedAttention(nn.Module):
    def __init__(self, d_model, num_heads, dilation_rate, segment_size, dropout=0.0, casual=False, use_xpos=False, use_rel_pos_bias=False, Distributed=False):
        super(DilatedAttention, self).__init__()

        #checking put parameter types
        assert isinstance(d_model, int) and d_model > 0, 'd_model should be positive integer'
        assert isinstance(num_heads, int) and num_heads > 0, 'num_heads should be positive integer'
        assert isinstance(dilation_rate, int) and dilation_rate > 0, 'dilation_rate should be a positive integer'
        
        assert isinstance(segment_size, int) and segment_size > 0, 'segment_size should be a positive integer'
        assert isinstance(dropout, float) and 0.0 <= dropout <= 1.0, 'dropout should be a positive integer'
        assert isinstance(casual, bool), 'casual should be a boolean value'

        assert isinstance(use_xpos, bool), 'use_xpos should be a boolean value '
        assert isinstance(use_rel_pos_bias, bool), 'use_rel_pos_bias should be a boolean value'
        
        # Initialize parameters
        self.d_model = d_model               # model dimension
        self.num_heads = num_heads           # number of attention heads
        self.dilation_rate = dilation_rate   # dilation rate
        self.segment_size = segment_size     # segment size
        self.dropout = nn.Dropout(dropout)
        # If casual attention is used
        self.casual = casual
        # If using positional encoding
        self.use_xpos = use_xpos
        # If using relative positional bias
        self.use_rel_pos_bias = use_rel_pos_bias
        self.distributed = Distributed


        # Initialize attention for each head with dilation
        if self.distributed:
            self.attentions = nn.ModuleList([DataParallel(FlashAttention(causal=self.casual, dropout=dropout)) for _ in range(self.dilation_rate)])
        else:
            self.attentions = nn.ModuleList([FlashAttention(causal=self.casual, dropout=dropout) for _ in range(self.dilation_rate)])

        # If using positional encoding, initialize it
        if use_xpos:
            self.xpos = XPOS(head_dim=d_model//num_heads)

        # If using relative positional bias, initialize it
        if use_rel_pos_bias:
            self.relative_bias = RelativePositionBias(num_buckets=32, max_distance=128, n_heads=num_heads)

        # Initialize softmax for later use in weights
        self.softmax = nn.Softmax(dim=-1)

    # Function to get mask for casual attention
    def get_mask(self, i, j):
        return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 2)


    # Forward function
    def forward(self, x):
        # Get batch size, sequence length and model dimension
        batch_size, seq_len, _ = x.shape

        # If using positional encoding, add it
        if self.use_xpos:
            x = self.xpos(x)

        # Initialize list to store outputs from each attention head
        all_head_outputs = []
        
        # For each attention head
        for head_idx, attention in enumerate(self.attentions):
            # Calculate offset for this head
            offset = head_idx % self.dilation_rate

            # Apply offset and segment for this head
            x_ = x[:, offset::self.dilation_rate, :]
            x_ = x_.contiguous().view(batch_size, -1, self.segment_size, self.d_model)
            

            
            elements_attns = []
            for idx in range(x_.shape[1]):
                element      = x_[:, idx, :, :].to(dtype)
                element_attn = attention(element, element, element)

                elements_attns.append(element_attn)

            attn_output = torch.cat(elements_attns, dim=1)


            #option2
            # elements_attns = [attention(element.to(dtype), element.to(dtype), element.to(dtype)) for element in x_]
            # attn_output = torch.cat(elements_attns, dim=1)
            
            # If using relative positional bias, add it
            if self.use_rel_pos_bias:
                attn_output += self.relative_bias(batch_size, attn_output.size(1), attn_output.size(1))

            # If using casual attention, apply mask
            if self.casual:
                mask = self.get_mask(attn_output.size(1), attn_output.size(1))
                attn_output = attn_output.masked_fill(mask, float('-inf'))

            # Apply dropout
            attn_output = self.dropout(attn_output)

            # Resize back to original size
            attn_output_resized = torch.zeros((batch_size, seq_len, self.d_model), device=device, dtype=dtype)
            attn_output_resized[:, offset::self.dilation_rate, :] = attn_output.contiguous().view(batch_size, -1, self.d_model)
            
            # Append output to list of all outputs
            all_head_outputs.append(attn_output_resized)

        # Calculate the weights for the different dilated attentions
        weights = self.softmax(torch.tensor([1.0 / self.dilation_rate for _ in range(self.dilation_rate)], device=device, dtype=dtype))

        # Apply the weights to the outputs of the different heads
        outputs_weighted = sum(w * out for w, out in zip(weights, all_head_outputs))

        # Return the weighted outputs
        return outputs_weighted



class MultiHeadDilatedAttention:
    def __init__():
        pass







class LongNetTransformer(nn.Module):
    def __init__(self, d_model, num_heads, dilation_rates, segment_sizes):
        super(LongNetTransformer, self).__init__()
        assert len(dilation_rates) == len(segment_sizes), "dilation_rates and segment_sizes should have the same length"


        self.d_model = d_model
        self.num_heads = num_heads
        self.dilation_rates = dilation_rates
        self.segment_sizes = segment_sizes
        
        self.dilated_attention_layers = nn.ModuleList(
            [DilatedAttention(d_model, num_heads, dilation_rate, segment_size)]
            for dilation_rate, segment_size in zip(dilation_rates, segment_sizes)
        )

    def forward(self, x):
        #accumlate outputs from different layers
        outputs = []

        #process each dilated attention layer
        for i in range(len(self.dilated_attention_layers)):
            output = self.dilated_attention_layers[i](x)
            outputs.append(output)

        #combine the outputs
        output = torch.sum(torch.stack(outputs), dim=0)

        return output
    