import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import clip
from model import CLIPVAD


class CLIPVADDescription(CLIPVAD):
    """VadCLIP variant that exposes video features for description alignment."""

    def __init__(
        self,
        classes_num,
        embed_dim,
        visual_length,
        visual_width,
        visual_head,
        visual_layers,
        attn_window,
        prompt_prefix,
        prompt_postfix,
        device,
    ):
        super().__init__(
            classes_num,
            embed_dim,
            visual_length,
            visual_width,
            visual_head,
            visual_layers,
            attn_window,
            prompt_prefix,
            prompt_postfix,
            device,
        )
        self.desc_projection = nn.Sequential(
            nn.Linear(visual_width, visual_width),
            nn.LayerNorm(visual_width),
            nn.GELU(),
            nn.Linear(visual_width, embed_dim),
        )

    def encode_description(self, descriptions):
        word_tokens = clip.tokenize(descriptions, truncate=True).to(self.device)
        word_embeddings = self.clipmodel.encode_token(word_tokens)
        return self.clipmodel.encode_text(word_embeddings, word_tokens)

    def forward(self, visual, padding_mask, text, lengths, return_visual=False):
        visual_features = self.encode_video(visual, padding_mask, lengths)
        logits1 = self.classifier(visual_features + self.mlp2(visual_features))

        text_features_ori = self.encode_textprompt(text)

        logits_attn = logits1.permute(0, 2, 1)
        visual_attn = logits_attn @ visual_features
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)
        visual_attn = visual_attn.expand(visual_attn.shape[0], text_features_ori.shape[0], visual_attn.shape[2])
        text_features = text_features_ori.unsqueeze(0)
        text_features = text_features.expand(visual_attn.shape[0], text_features.shape[1], text_features.shape[2])
        text_features = text_features + visual_attn
        text_features = text_features + self.mlp1(text_features)

        visual_features_norm = F.normalize(visual_features, dim=-1)
        text_features_norm = F.normalize(text_features, dim=-1).permute(0, 2, 1)
        logits2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07

        if return_visual:
            return text_features_ori, logits1, logits2, visual_features
        return text_features_ori, logits1, logits2
