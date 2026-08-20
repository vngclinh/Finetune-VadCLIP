import math

import torch
import torch.nn.functional as F

from model import CLIPVAD


class CLIPVADMultiPrompt(CLIPVAD):
    def encode_textprompt_groups(self, prompt_groups):
        if not prompt_groups:
            raise ValueError("prompt_groups must not be empty.")
        if isinstance(prompt_groups[0], str):
            grouped = self.encode_textprompt(prompt_groups).unsqueeze(1)
            return grouped

        group_sizes = [len(group) for group in prompt_groups]
        if any(size == 0 for size in group_sizes):
            raise ValueError("Each prompt group must contain at least one prompt.")
        if len(set(group_sizes)) != 1:
            raise ValueError(f"All prompt groups must have the same size. Got: {sorted(set(group_sizes))}")

        flat_prompts = [prompt for group in prompt_groups for prompt in group]
        flat_features = self.encode_textprompt(flat_prompts)
        return flat_features.reshape(len(prompt_groups), group_sizes[0], flat_features.shape[-1])

    @staticmethod
    def aggregate_group_features(grouped_features, prompt_weights=None):
        original_dtype = grouped_features.dtype
        grouped_features = F.normalize(grouped_features.float(), dim=-1)
        if prompt_weights is None:
            return F.normalize(grouped_features.mean(dim=1), dim=-1).to(original_dtype)

        weights = prompt_weights.to(grouped_features.device, dtype=grouped_features.dtype)
        if weights.shape != grouped_features.shape[:2]:
            raise ValueError(
                f"prompt_weights shape {tuple(weights.shape)} does not match grouped_features "
                f"shape {tuple(grouped_features.shape[:2])}"
            )
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        weighted_features = (grouped_features * weights.unsqueeze(-1)).sum(dim=1)
        return F.normalize(weighted_features, dim=-1).to(original_dtype)

    def forward(
        self,
        visual,
        padding_mask,
        text,
        lengths,
        return_visual=False,
        return_prompt_logits=False,
        prompt_weights=None,
    ):
        visual_features = self.encode_video(visual, padding_mask, lengths)
        logits1 = self.classifier(visual_features + self.mlp2(visual_features))

        text_features_grouped_ori = self.encode_textprompt_groups(text)
        text_features_ori = self.aggregate_group_features(text_features_grouped_ori, prompt_weights)

        logits_attn = logits1.permute(0, 2, 1)
        visual_attn = logits_attn @ visual_features
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        visual_attn = visual_attn.unsqueeze(2).expand(
            visual_attn.shape[0],
            text_features_grouped_ori.shape[0],
            text_features_grouped_ori.shape[1],
            visual_attn.shape[-1],
        )

        text_features = text_features_grouped_ori.unsqueeze(0).expand(
            visual_attn.shape[0],
            text_features_grouped_ori.shape[0],
            text_features_grouped_ori.shape[1],
            text_features_grouped_ori.shape[2],
        )
        text_features = text_features + visual_attn
        text_features = text_features + self.mlp1(text_features)

        visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        prompt_logits = torch.einsum(
            "btd,bckd->btck",
            visual_features_norm,
            text_features_norm.type(visual_features_norm.dtype),
        ) / 0.07
        if prompt_weights is None:
            logits2 = torch.logsumexp(prompt_logits, dim=-1) - math.log(prompt_logits.shape[-1])
        else:
            weights = prompt_weights.to(prompt_logits.device, dtype=prompt_logits.dtype)
            if weights.shape != prompt_logits.shape[2:4]:
                raise ValueError(
                    f"prompt_weights shape {tuple(weights.shape)} does not match prompt logits "
                    f"class/prompt shape {tuple(prompt_logits.shape[2:4])}"
                )
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            logits2 = torch.logsumexp(prompt_logits + weights.clamp_min(1e-8).log().view(1, 1, *weights.shape), dim=-1)

        outputs = [text_features_ori, logits1, logits2]
        if return_prompt_logits:
            outputs.append(prompt_logits)
        if return_visual:
            outputs.append(visual_features)
        return tuple(outputs)
