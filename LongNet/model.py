import torch
from torch.nn import Embedding, Module
import bitsandbytes

from LongNet.torchscale import DecoderConfig, Decoder, PositionalEmbedding
from transformers import AutoTokenizer

# 
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


class LongNet(Module):
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
        

# class LongNetSelector:
#     def __init__(self, mode="language"):
#         assert mode in ['multimodal', 'language'], 'Invalid mode choose from multimodal or language'
#         if mode == "multimodal":
#             self.model = LongNet()
#         else:
#             self.model = LongNetLanguage()


#     def forward(self, *args, **kwargs):
#         return self.model(*args, **kwargs)