import json

import torch


COMPACT_PROTOTYPE_TYPES = ["keyword_phrase", "compact_action_phrase", "short_visual_sentence"]
LONG_EVENT_PROTOTYPE_TYPES = ["core_action", "actor_object_interaction", "temporal_progression"]


def _format_prompt(class_text, prototype_text, prompt_mode):
    if prompt_mode == "prototype_only":
        return prototype_text
    if prompt_mode == "manual_caption":
        return f"{class_text} {prototype_text}"
    if prompt_mode in ("class_plus_prototype", "class_anchor"):
        return f"{class_text} involving {prototype_text}"
    raise ValueError(f"Unknown prompt mode: {prompt_mode}")


def build_compact_prompt_groups(prototype_json, label_map, prompt_replacement_mode, compact_prototype_types):
    with open(prototype_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    classes = payload.get("classes", {})
    prompt_groups = []
    for raw_label, class_text in label_map.items():
        class_payload = classes.get(raw_label)
        if not class_payload:
            raise KeyError(f"Missing compact prototypes for class: {raw_label}")

        compact_prototypes = class_payload.get("compact_prototypes", [])
        by_type = {
            item.get("type"): item.get("text", "").strip()
            for item in compact_prototypes
            if item.get("type") and item.get("text", "").strip()
        }
        missing_types = [prototype_type for prototype_type in compact_prototype_types if prototype_type not in by_type]
        if missing_types:
            raise ValueError(
                f"Missing compact prototype types for {raw_label}: {missing_types}. "
                f"Available types: {sorted(by_type)}"
            )

        group = []
        for prototype_type in compact_prototype_types:
            prototype_text = by_type[prototype_type]
            group.append(_format_prompt(class_text, prototype_text, prompt_replacement_mode))
        prompt_groups.append(group)

    return prompt_groups


def build_multi_prompt_groups(
    prototype_json,
    label_map,
    prototype_source,
    compact_prototype_types=None,
    long_prototype_types=None,
    prompt_mode="class_plus_prototype",
):
    with open(prototype_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    classes = payload.get("classes", {})
    compact_prototype_types = compact_prototype_types or COMPACT_PROTOTYPE_TYPES
    long_prototype_types = long_prototype_types or LONG_EVENT_PROTOTYPE_TYPES

    prompt_groups = []
    for raw_label, class_text in label_map.items():
        class_payload = classes.get(raw_label)
        if not class_payload:
            raise KeyError(f"Missing prototypes for class: {raw_label}")

        if prototype_source == "manual":
            prompts = class_payload
            if not isinstance(prompts, list):
                raise ValueError(f"Manual prompt source expects a list of prompts for {raw_label}.")
            group = [prompt.strip() for prompt in prompts if prompt.strip()]
            if not group:
                raise ValueError(f"Manual prompt source has no prompts for {raw_label}.")
            prompt_groups.append([_format_prompt(class_text, prompt, prompt_mode) for prompt in group])
            continue

        if prototype_source == "compact":
            prototypes = class_payload.get("compact_prototypes", [])
            selected_types = compact_prototype_types
            source_name = "compact_prototypes"
        elif prototype_source in ("long", "long_event", "long_event_only"):
            prototypes = class_payload.get("final_prototypes", [])
            selected_types = long_prototype_types
            source_name = "final_prototypes"
        else:
            raise ValueError(f"Unknown prototype_source: {prototype_source}")

        by_type = {
            item.get("type"): item.get("text", "").strip()
            for item in prototypes
            if item.get("type") and item.get("text", "").strip()
        }
        missing_types = [prototype_type for prototype_type in selected_types if prototype_type not in by_type]
        if missing_types:
            raise ValueError(
                f"Missing {source_name} types for {raw_label}: {missing_types}. "
                f"Available types: {sorted(by_type)}"
            )

        group = [
            _format_prompt(class_text, by_type[prototype_type], prompt_mode)
            for prototype_type in selected_types
        ]
        prompt_groups.append(group)

    group_sizes = {len(group) for group in prompt_groups}
    if len(group_sizes) != 1:
        raise ValueError(f"All classes must have the same number of prompts. Got sizes: {sorted(group_sizes)}")

    return prompt_groups


def build_multi_prompt_weights(prompt_weights_json, label_map, device=None):
    if not prompt_weights_json:
        return None

    with open(prompt_weights_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    weights_by_class = payload.get("prompt_weights")
    if not isinstance(weights_by_class, dict):
        raise ValueError(f"Missing prompt_weights object in: {prompt_weights_json}")

    rows = []
    group_sizes = set()
    for raw_label in label_map:
        weights = weights_by_class.get(raw_label)
        if weights is None:
            raise KeyError(f"Missing prompt weights for class: {raw_label}")
        if not isinstance(weights, list) or not weights:
            raise ValueError(f"Prompt weights for {raw_label} must be a non-empty list.")
        row = [float(value) for value in weights]
        if any(value < 0 for value in row):
            raise ValueError(f"Prompt weights for {raw_label} must be non-negative.")
        total = sum(row)
        if total <= 0:
            raise ValueError(f"Prompt weights for {raw_label} must sum to a positive value.")
        row = [value / total for value in row]
        rows.append(row)
        group_sizes.add(len(row))

    if len(group_sizes) != 1:
        raise ValueError(f"All classes must have the same number of prompt weights. Got sizes: {sorted(group_sizes)}")

    tensor = torch.tensor(rows, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor
