import torch
from torch.nn import Embedding, Module
import bitsandbytes

from transformers import AutoTokenizer


from torchscale.architecture.config import DecoderConfig
from torchscale.architecture.decoder import Decoder
from torchscale.component.embedding import PositionalEmbedding


from LongNet.Transformer import LongNet

class LongNetTokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neox-20b",
            eos_token="<eos>",
            pad_token="<pad>",
            extra_ids=0,
            model_max_length=8192
        )

    def tokenize_texts(self, texts):
        return self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).input_ids


class LongNetTorchscale(Module):
    def __init__(self):
        super().__init__()
        self.embed = bitsandbytes.nn.modules.Embedding(
            320002,
            2048,
            padding_idx=1
        )

        self.embed_positions = PositionalEmbedding(
            2048,
            2048,
            1
        )

        self.output_projection = torch.nn.Linear(
            2048, 32002, bias=False
        )

        self.config = DecoderConfig(
            decoder_layers=24,
            decoder_embed_dim=2048,
            decoder_ffn_embed_dim=8192,
            decoder_attention_heads=32,
            dropout=0.1,
            activation_fn="gelu",
            attention_dropout=0.1,
            decoder_dilation_rate=4,
            decoder_segment_size=2,
            vocab_size=64007,
        )

        self.decoder = Decoder(
            self.config,
            embed_tokens=self.embed,
            embed_positions=self.embed_positions,
            output_projection=self.output_projection
        )


    def forward(self, text_tokens, **kwargs):
        model_input = self.decoder.forward_embedding(text_tokens)[0]
        return self.decoder(model_input, passed_x=model_input)[0]
        

class DilatedLongNet(Module):
    def __init__(self):
        super().__init__()

        self.model = LongNet(
            num_tokens = 16000,             # number of tokens
            dim = (512, 256),               # transformer model dimension (512 for coarsest, 256 for fine in this example)
            max_seq_len = (1024, 4),        # sequence length for global and then local. this can be more than 2
            depth = (6, 4),                 # number of layers for global and then local. this can be more than 2, but length must match the max_seq_len's
            dim_head = 64,                  # dimension per head
            heads = 8,                      # number of attention heads
            flash_attn = True,              # use flash attention
            dilation_rate = 1,              # dilation rate for DilatedAttention
            segment_size = 0,               # segment size for DilatedAttention
            casual = False,                 # whether to use causal attention for DilatedAttention
            use_xpos = False,               # whether to use absolute positional embeddings for DilatedAttention
            use_rel_pos_bias = False,       # whether to use relative positional bias for DilatedAttention
            distributed = False             # whether to distribute attention for DilatedAttention
        )

    def generate(self, text_tokens, temperature: int = None, filter_thres: int = None, **kwargs):
        sampled = self.model.generate(temperature=temperature, filter_thres=filter_thres)
        return sampled

    