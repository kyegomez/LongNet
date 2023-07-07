import torch
from LongNet import DecoderConfig, Decoder

from torchscale.component.embedding import PositionalEmbedding
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
from flamingo_pytorch import PerceiverResampler

from torch.nn import Embedding, Module
import bitsandbytes

class LongNetTokenizer:
    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neox-20b",
            additional_special_tokens=["<image>", "</image>"],
            eos_token="<eos>",
            pad_token="<pad>",
            extra_ids=0,
            model_max_length=8192
        )
        self.im_idx, self.im_end_idx = self.tokenizer.convert_tokens_to_ids(["<image>", "</image>"])

    def tokenize_texts(self, texts):
        texts = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).input_ids
        image_tokens = torch.tensor([[self.im_idx, self.im_end_idx]] * texts.shape[0])
        return torch.cat([texts[:, 0:1], image_tokens, texts[:, 1:]], dim=1), texts

    def tokenize_images(self, images):
        return self.processor(images=images, return_tensors="pt").pixel_values

    def tokenize(self, sample):
        text_tokens, only_text_tokens = self.tokenize_texts(sample["target_text"])
        attention_mask = text_tokens != self.tokenizer.pad_token_id
        dummy_image_features = torch.ones((text_tokens.shape[0], 64))
        attention_mask = torch.cat([dummy_image_features, attention_mask], dim=1)
        return {
            "text_tokens": text_tokens,
            "images": self.tokenize_images(sample["image"]),
            "labels": only_text_tokens,
            "attention_mask": attention_mask,
        }

class LongNet(Module):
    def __init__(self):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K").vision_model

        self.embed = bitsandbytes.nn.modules.Embedding(
            32002,
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
        torch.nn.init.normal_(
            self.output_projection.weight, mean=0, std=2048**-0.5
        )

        self.config = DecoderConfig(
            decoder_layers=24,
            decoder_embed_dim=2048,
            decoder_ffn_embed_dim=8192,
            decoder_attention_heads=32,
            dropout=0.1,
            activation_fn="gelu",
            attention_dropout=0.1,
            decoder_dilation_rate=4,    # New: Dilation Rate
            decoder_segment_size=2,     # New: Segment Size
            vocab_size=64007,
            # subln=True,
            # xpos_rel_pos=True,
            # multiway=True,
            # max_rel_pos=2048,
            # alibi_pos_bias=True,
            # alibi_num_heads=16,
        )
        self.decoder = Decoder(
            self.config,
            embed_tokens=self.embed,
            embed_positions=self.embed_positions,
            output_projection=self.output_projection
        )

        self.perceive = PerceiverResampler(
            dim=1024,
            depth=2,
            dim_head=64,
            heads=8,
            num_latents=64,
            num_media_embeds=257
        )

        self.image_proj = torch.nn.Linear(1024, 2048, bias=False)
        torch.nn.init.normal_(
            self.image_proj.weight, mean=0, std=2048**-0.5
        )

    def forward(self, text_tokens, images, **kwargs):
        images = self.clip_model(pixel_values=images)["last_hidden_state"]
        images = self.perceive(images).squeeze(1)
        images = self.image_proj(images)

        model_input = self.decoder.forward_embedding(text_tokens)[1]
        model_input = torch.cat([model_input[:, 0:2], images, model_input[:, 2:]], dim=1)
        model_input = self.decoder.forward_embedding(model_input, token_embedding=model_input)[0]

        return self.decoder(model_input, passed_x=model_input)[0]

class LongNetLanguage(Module):
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
        

class LongNetSelector:
    def __init__(self, mode="language"):
        assert mode in ['multimodal', 'language'], 'Invalid mode choose from multimodal or language'
        if mode == "multimodal":
            self.model = LongNet()
        else:
            self.model = LongNetLanguage()


    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)