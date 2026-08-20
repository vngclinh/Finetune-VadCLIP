import torch
import torch.nn.functional as F

from model import CLIPVAD


class CLIPVADPromptReplacement(CLIPVAD):
    def encode_textprompt_groups(self, prompt_groups):
        if not prompt_groups:
            raise ValueError("prompt_groups must not be empty.")
        if isinstance(prompt_groups[0], str):
            return self.encode_textprompt(prompt_groups)

        flat_prompts = []
        group_sizes = []
        for group in prompt_groups:
            if not group:
                raise ValueError("Each prompt group must contain at least one prompt.")
            flat_prompts.extend(group)
            group_sizes.append(len(group))

        flat_features = self.encode_textprompt(flat_prompts)
        grouped_features = []
        cursor = 0
        for group_size in group_sizes:
            group_features = F.normalize(flat_features[cursor: cursor + group_size].float(), dim=-1)
            centroid = F.normalize(group_features.mean(dim=0, keepdim=True), dim=-1).squeeze(0)
            grouped_features.append(centroid.to(flat_features.dtype))
            cursor += group_size

        return torch.stack(grouped_features, dim=0)

    def forward(self, visual, padding_mask, text, lengths, return_visual=False):
        visual_features = self.encode_video(visual, padding_mask, lengths)
        logits1 = self.classifier(visual_features + self.mlp2(visual_features))

        text_features_ori = self.encode_textprompt_groups(text)

        logits_attn = logits1.permute(0, 2, 1)
        visual_attn = logits_attn @ visual_features
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)
        visual_attn = visual_attn.expand(visual_attn.shape[0], text_features_ori.shape[0], visual_attn.shape[2])
        text_features = text_features_ori.unsqueeze(0)
        text_features = text_features.expand(visual_attn.shape[0], text_features.shape[1], text_features.shape[2])
        text_features = text_features + visual_attn
        text_features = text_features + self.mlp1(text_features)

        visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features_norm.permute(0, 2, 1)
        logits2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07

        if return_visual:
            return text_features_ori, logits1, logits2, visual_features

        return text_features_ori, logits1, logits2
