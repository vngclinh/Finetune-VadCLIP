import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "VadCLIP" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clip import clip  # noqa: E402
from model import CLIPVAD  # noqa: E402
from utils.prompt_replacement import build_multi_prompt_groups  # noqa: E402


PROMPT_LABEL_MAP = {
    "Normal": "normal",
    "Abuse": "abuse",
    "Arrest": "arrest",
    "Arson": "arson",
    "Assault": "assault",
    "Burglary": "burglary",
    "Explosion": "explosion",
    "Fighting": "fighting",
    "RoadAccidents": "roadAccidents",
    "Robbery": "robbery",
    "Shooting": "shooting",
    "Shoplifting": "shoplifting",
    "Stealing": "stealing",
    "Vandalism": "vandalism",
}


MODEL_CONFIG = dict(
    num_class=14,
    embed_dim=512,
    visual_length=256,
    visual_width=512,
    visual_head=1,
    visual_layers=2,
    attn_window=8,
    prompt_prefix=10,
    prompt_postfix=10,
)


def discover_prompt_jsons():
    prompt_files = []
    for pattern in ("ucf_manual_class_prompts_5x*.json", "ucf_manual_class_prompts_10x*.json"):
        prompt_files.extend(PROJECT_ROOT.joinpath("code").glob(pattern))
    prompt_files = [
        path for path in prompt_files
        if "selected" not in path.stem and "representative" not in path.stem
    ]
    return sorted({path.resolve() for path in prompt_files})


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Score UCF-Crime prompt banks in VadCLIP/CLIP text space and select "
            "one representative prompt per class."
        )
    )
    parser.add_argument(
        "--prompt-jsons",
        nargs="+",
        default=None,
        help="Prompt-bank JSON files. If omitted, all code/ucf_manual_class_prompts_5x*.json and 10x*.json are used.",
    )
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "model_ucf.pth"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "code" / "vadclip_representative_prompt_selection"))
    parser.add_argument(
        "--selected-output-json",
        default=str(PROJECT_ROOT / "code" / "ucf_selected_representative_class_prompts.json"),
        help="Main selected prompt bank for training, produced from --main-space.",
    )
    parser.add_argument(
        "--weighted-output-json",
        default=str(PROJECT_ROOT / "code" / "ucf_weighted_class_prompts.json"),
        help=(
            "Main weighted prompt bank for training. Written when --main-space is analyzed; "
            "if multiple prompt JSONs are provided, the first input bank is used for this file "
            "and all banks are also written under the output directory."
        ),
    )
    parser.add_argument(
        "--prompt-weight-temperature",
        default=0.20,
        type=float,
        help="Softmax temperature used to convert per-class prompt scores into prompt weights.",
    )
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--w-anchor", default=0.50, type=float)
    parser.add_argument("--w-intra", default=0.30, type=float)
    parser.add_argument("--w-inter", default=0.20, type=float)
    parser.add_argument(
        "--inter-mode",
        default="max",
        choices=["max", "top3_mean", "mean"],
        help="How to aggregate similarity to prompts from other classes.",
    )
    parser.add_argument(
        "--main-space",
        default="raw_clip_encode_text",
        choices=["vadclip_encode_textprompt", "raw_clip_encode_text"],
        help="Embedding space used to write --selected-output-json.",
    )
    parser.add_argument(
        "--scoring-prompt-mode",
        default="prototype_only",
        choices=["manual_caption", "class_plus_prototype", "prototype_only"],
        help=(
            "How prompt candidates are formatted before scoring. prototype_only measures "
            "the raw prompt against class-name anchors without injecting class names."
        ),
    )
    parser.add_argument(
        "--spaces",
        nargs="+",
        default=["vadclip_encode_textprompt", "raw_clip_encode_text"],
        choices=["vadclip_encode_textprompt", "raw_clip_encode_text"],
    )
    return parser.parse_args()


def camel_to_words(text):
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    return text.replace("_", " ").strip().lower()


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print("Saved CSV:", path)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved JSON:", path)


def normalize_np(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def instantiate_model(model_path, device):
    config = dict(MODEL_CONFIG)
    config["device"] = device
    model = CLIPVAD(**config)
    state_dict = torch.load(model_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print("Loaded VadCLIP text encoder checkpoint:", model_path)
    print("  missing keys:", len(missing), "| unexpected keys:", len(unexpected))
    return model


def encode_texts_with_vadclip(model, texts, batch_size):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            emb = model.encode_textprompt(batch)
            emb = F.normalize(emb.float(), dim=-1)
            chunks.append(emb.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def encode_texts_with_raw_clip(model, texts, batch_size, device):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            tokens = clip.tokenize(batch, truncate=True).to(device)
            token_embeddings = model.clipmodel.encode_token(tokens)
            emb = model.clipmodel.encode_text(token_embeddings, tokens)
            emb = F.normalize(emb.float(), dim=-1)
            chunks.append(emb.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def encode_texts(model, texts, space_name, batch_size, device):
    if space_name == "vadclip_encode_textprompt":
        return encode_texts_with_vadclip(model, texts, batch_size)
    if space_name == "raw_clip_encode_text":
        return encode_texts_with_raw_clip(model, texts, batch_size, device)
    raise ValueError(f"Unknown embedding space: {space_name}")


def get_clip_token_rows(texts, candidate_rows):
    tokens = clip.tokenize(texts, truncate=True)
    rows = []
    for i, base in enumerate(candidate_rows):
        nonzero_ids = [int(token_id) for token_id in tokens[i].tolist() if int(token_id) != 0]
        row = dict(base)
        row.update(
            {
                "global_prompt_index": i,
                "clip_token_count_with_special": len(nonzero_ids),
                "clip_content_token_count": max(0, len(nonzero_ids) - 2),
                "clip_token_ids_nonzero": " ".join(str(token_id) for token_id in nonzero_ids),
            }
        )
        rows.append(row)
    return rows


def load_prompt_bank(prompt_json, scoring_prompt_mode):
    prompt_json = Path(prompt_json).resolve()
    payload = json.loads(prompt_json.read_text(encoding="utf-8"))
    prompt_groups = build_multi_prompt_groups(
        str(prompt_json),
        PROMPT_LABEL_MAP,
        "manual",
        prompt_mode=scoring_prompt_mode,
    )
    raw_classes = payload.get("classes", {})
    class_names = list(PROMPT_LABEL_MAP.keys())
    group_sizes = sorted({len(group) for group in prompt_groups})
    if len(group_sizes) != 1:
        raise ValueError(f"Prompt groups must have equal sizes in {prompt_json}. Got: {group_sizes}")

    rows = []
    encoded_prompts = []
    for class_index, class_name in enumerate(class_names):
        if class_name not in raw_classes:
            raise KeyError(f"Missing class {class_name} in {prompt_json}")
        raw_group = raw_classes[class_name]
        for prompt_index, encoded_prompt in enumerate(prompt_groups[class_index]):
            raw_prompt = raw_group[prompt_index].strip()
            encoded_prompts.append(encoded_prompt)
            rows.append(
                {
                    "source_file": prompt_json.name,
                    "source_stem": prompt_json.stem,
                    "source_prompts_per_class": group_sizes[0],
                    "class_index": class_index,
                    "class_name": class_name,
                    "class_anchor_text": PROMPT_LABEL_MAP[class_name],
                    "prompt_index": prompt_index,
                    "raw_prompt_text": raw_prompt,
                    "encoded_prompt_text": encoded_prompt,
                }
            )
    return payload, encoded_prompts, rows


def load_candidate_pool(prompt_jsons, scoring_prompt_mode):
    all_payloads = {}
    all_texts = []
    all_rows = []
    for prompt_json in prompt_jsons:
        payload, texts, rows = load_prompt_bank(prompt_json, scoring_prompt_mode)
        all_payloads[Path(prompt_json).name] = payload.get("metadata", {})
        offset = len(all_texts)
        for local_index, row in enumerate(rows):
            row = dict(row)
            row["global_prompt_index"] = offset + local_index
            all_rows.append(row)
        all_texts.extend(texts)
    return all_payloads, all_texts, all_rows


def aggregate_inter_similarity(other_scores, mode):
    if mode == "max":
        return float(other_scores.max())
    if mode == "top3_mean":
        top_k = min(3, len(other_scores))
        return float(np.sort(other_scores)[-top_k:].mean())
    if mode == "mean":
        return float(other_scores.mean())
    raise ValueError(f"Unknown inter mode: {mode}")


def rank_candidates(prompt_embeddings, anchor_embeddings, candidate_rows, class_names, weights, inter_mode):
    prompt_similarity = prompt_embeddings @ prompt_embeddings.T
    prompt_to_anchor = prompt_embeddings @ anchor_embeddings.T
    anchor_similarity = anchor_embeddings @ anchor_embeddings.T

    indices_by_class = {
        class_index: [i for i, row in enumerate(candidate_rows) if row["class_index"] == class_index]
        for class_index in range(len(class_names))
    }
    scored_rows = []
    for row_index, base in enumerate(candidate_rows):
        class_index = base["class_index"]
        same_indices = [idx for idx in indices_by_class[class_index] if idx != row_index]
        other_indices = [
            idx for other_class, indices in indices_by_class.items()
            if other_class != class_index
            for idx in indices
        ]

        intra_similarity = float(prompt_similarity[row_index, same_indices].mean()) if same_indices else 1.0
        inter_similarity = aggregate_inter_similarity(prompt_similarity[row_index, other_indices], inter_mode)
        anchor_similarity_own = float(prompt_to_anchor[row_index, class_index])
        selection_score = (
            weights["anchor"] * anchor_similarity_own
            + weights["intra"] * intra_similarity
            - weights["inter"] * inter_similarity
        )

        anchor_scores = prompt_to_anchor[row_index]
        best_other_anchor = max(
            [idx for idx in range(len(class_names)) if idx != class_index],
            key=lambda idx: anchor_scores[idx],
        )
        scored = dict(base)
        scored.update(
            {
                "anchor_similarity": anchor_similarity_own,
                "intra_prompt_similarity": intra_similarity,
                "inter_prompt_similarity": inter_similarity,
                "selection_score": float(selection_score),
                "best_other_anchor_class": class_names[best_other_anchor],
                "best_other_anchor_similarity": float(anchor_scores[best_other_anchor]),
                "anchor_margin": float(anchor_similarity_own - anchor_scores[best_other_anchor]),
            }
        )
        scored_rows.append(scored)

    selected_rows = []
    for class_index, class_name in enumerate(class_names):
        class_rows = [row for row in scored_rows if row["class_index"] == class_index]
        class_rows.sort(key=lambda row: row["selection_score"], reverse=True)
        for rank, row in enumerate(class_rows, start=1):
            row["rank_in_class"] = rank
        selected_rows.append(dict(class_rows[0]))
    scored_rows.sort(key=lambda row: (row["class_index"], row["rank_in_class"]))

    selected_indices = [row["global_prompt_index"] for row in selected_rows]
    selected_embeddings = prompt_embeddings[selected_indices]
    selected_similarity = selected_embeddings @ selected_embeddings.T
    return scored_rows, selected_rows, prompt_similarity, prompt_to_anchor, anchor_similarity, selected_similarity


def softmax_np(values, temperature):
    values = np.asarray(values, dtype=np.float64)
    temperature = max(float(temperature), 1e-6)
    scaled = values / temperature
    scaled = scaled - scaled.max()
    exp_values = np.exp(scaled)
    return exp_values / np.maximum(exp_values.sum(), 1e-12)


def add_prompt_weights(scored_rows, temperature):
    rows = [dict(row) for row in scored_rows]
    groups = {}
    for row in rows:
        key = (row["source_file"], row["class_index"])
        groups.setdefault(key, []).append(row)

    for group_rows in groups.values():
        group_rows.sort(key=lambda row: row["prompt_index"])
        weights = softmax_np([row["selection_score"] for row in group_rows], temperature)
        rank_order = sorted(range(len(group_rows)), key=lambda idx: weights[idx], reverse=True)
        ranks = {idx: rank + 1 for rank, idx in enumerate(rank_order)}
        for idx, row in enumerate(group_rows):
            row["prompt_weight"] = float(weights[idx])
            row["weight_rank_in_class"] = ranks[idx]
            row["prompt_weight_temperature"] = float(temperature)
    return rows


def build_prompt_to_anchor_rows(scored_rows, prompt_to_anchor, class_names):
    rows = []
    for row_index, base in enumerate(scored_rows):
        row = {
            "global_prompt_index": base["global_prompt_index"],
            "source_file": base["source_file"],
            "source_stem": base["source_stem"],
            "class_index": base["class_index"],
            "class_name": base["class_name"],
            "prompt_index": base["prompt_index"],
            "raw_prompt_text": base["raw_prompt_text"],
            "encoded_prompt_text": base["encoded_prompt_text"],
            "own_class_similarity": base["anchor_similarity"],
            "best_other_anchor_class": base["best_other_anchor_class"],
            "best_other_anchor_similarity": base["best_other_anchor_similarity"],
            "anchor_margin": base["anchor_margin"],
        }
        for class_index, class_name in enumerate(class_names):
            row[f"sim_to_{class_name}"] = float(prompt_to_anchor[row_index, class_index])
        rows.append(row)
    return rows


def build_weighted_payload_for_source(
    source_file,
    weighted_rows,
    prompt_jsons,
    weights,
    inter_mode,
    space_name,
    scoring_prompt_mode,
    temperature,
):
    source_rows = [row for row in weighted_rows if row["source_file"] == source_file]
    if not source_rows:
        raise ValueError(f"No rows found for source_file={source_file}")

    class_names = list(PROMPT_LABEL_MAP.keys())
    classes = {}
    prompt_weights = {}
    scoring_details = {}
    for class_name in class_names:
        class_rows = [row for row in source_rows if row["class_name"] == class_name]
        class_rows.sort(key=lambda row: row["prompt_index"])
        classes[class_name] = [row["raw_prompt_text"] for row in class_rows]
        prompt_weights[class_name] = [row["prompt_weight"] for row in class_rows]
        scoring_details[class_name] = class_rows

    return {
        "metadata": {
            "source_prompt_json": source_file,
            "source_prompt_jsons_all_inputs": [str(Path(path).resolve()) for path in prompt_jsons],
            "selection_method": "anchor_intra_inter_text_space_score",
            "weighting_method": "softmax_per_class_over_selection_score",
            "embedding_space": space_name,
            "scoring_prompt_mode": scoring_prompt_mode,
            "training_prompt_mode": scoring_prompt_mode,
            "note": (
                "classes contain raw prompt text; prompt_weights has the same order. "
                "Use the same prompt_mode at training/evaluation unless intentionally testing a mismatch."
            ),
            "score_weights": weights,
            "inter_mode": inter_mode,
            "prompt_weight_temperature": float(temperature),
            "prompts_per_class": len(next(iter(classes.values()))),
        },
        "classes": classes,
        "prompt_weights": prompt_weights,
        "scoring_details": scoring_details,
    }


def compare_baseline_and_selected(class_names, anchor_similarity, selected_similarity):
    rows = []
    for i, class_name in enumerate(class_names):
        other_indices = [j for j in range(len(class_names)) if j != i]
        baseline_nearest = max(other_indices, key=lambda j: anchor_similarity[i, j])
        selected_nearest = max(other_indices, key=lambda j: selected_similarity[i, j])
        rows.append(
            {
                "class_name": class_name,
                "baseline_nearest_class": class_names[baseline_nearest],
                "baseline_nearest_similarity": float(anchor_similarity[i, baseline_nearest]),
                "baseline_separation": float(1.0 - anchor_similarity[i, baseline_nearest]),
                "selected_nearest_class": class_names[selected_nearest],
                "selected_nearest_similarity": float(selected_similarity[i, selected_nearest]),
                "selected_separation": float(1.0 - selected_similarity[i, selected_nearest]),
                "separation_delta_selected_minus_baseline": float(
                    (1.0 - selected_similarity[i, selected_nearest])
                    - (1.0 - anchor_similarity[i, baseline_nearest])
                ),
            }
        )
    return sorted(rows, key=lambda row: row["separation_delta_selected_minus_baseline"])


def build_selected_payload(prompt_jsons, selected_rows, weights, inter_mode, space_name, scoring_prompt_mode):
    return {
        "metadata": {
            "source_prompt_jsons": [str(Path(path).resolve()) for path in prompt_jsons],
            "selection_method": "anchor_intra_inter_text_space_score",
            "embedding_space": space_name,
            "scoring_prompt_mode": scoring_prompt_mode,
            "training_prompt_mode": scoring_prompt_mode,
            "note": "classes contain raw prompt text only; use the same prompt_mode at training/evaluation unless intentionally testing a mismatch.",
            "weights": weights,
            "inter_mode": inter_mode,
            "prompts_per_class": 1,
        },
        "classes": {row["class_name"]: [row["raw_prompt_text"]] for row in selected_rows},
        "selection_details": {row["class_name"]: row for row in selected_rows},
    }


def print_console_summary(space_name, selected_rows, scored_rows, baseline_rows):
    print()
    print("=" * 110)
    print("Embedding space:", space_name)
    print("Selected representative prompts:")
    for row in selected_rows:
        print(
            f"  {row['class_name']:<13} "
            f"{row['source_stem']}#{row['prompt_index']:<2} "
            f"score={row['selection_score']:+.4f} "
            f"anchor={row['anchor_similarity']:.4f} "
            f"intra={row['intra_prompt_similarity']:.4f} "
            f"inter={row['inter_prompt_similarity']:.4f} "
            f"margin={row['anchor_margin']:+.4f} | {row['raw_prompt_text']}"
        )

    wrong_anchor = [row for row in scored_rows if row["anchor_margin"] < 0]
    print()
    print("Prompts whose nearest class-name anchor is not their own:", len(wrong_anchor), "/", len(scored_rows))
    for row in sorted(wrong_anchor, key=lambda item: item["anchor_margin"])[:15]:
        print(
            f"  {row['class_name']} {row['source_stem']}#{row['prompt_index']} "
            f"own={row['anchor_similarity']:.4f} "
            f"best={row['best_other_anchor_class']}:{row['best_other_anchor_similarity']:.4f} "
            f"margin={row['anchor_margin']:+.4f} | {row['raw_prompt_text']}"
        )

    print()
    print("Lowest prompt-to-own-class similarity:")
    for row in sorted(scored_rows, key=lambda item: item["anchor_similarity"])[:25]:
        print(
            f"  {row['class_name']} {row['source_stem']}#{row['prompt_index']} "
            f"own_class_sim={row['anchor_similarity']:.4f} "
            f"best_other={row['best_other_anchor_class']}:{row['best_other_anchor_similarity']:.4f} "
            f"margin={row['anchor_margin']:+.4f} | {row['raw_prompt_text']}"
        )

    print()
    print("Lowest total-score prompts:")
    for row in sorted(scored_rows, key=lambda item: item["selection_score"])[:15]:
        print(
            f"  {row['class_name']} {row['source_stem']}#{row['prompt_index']} "
            f"score={row['selection_score']:+.4f} "
            f"anchor={row['anchor_similarity']:.4f} "
            f"intra={row['intra_prompt_similarity']:.4f} "
            f"inter={row['inter_prompt_similarity']:.4f} | {row['raw_prompt_text']}"
        )

    print()
    print("Text-space baseline class-name vs selected prompt separation:")
    for row in baseline_rows:
        print(
            f"  {row['class_name']:<13} "
            f"baseline->{row['baseline_nearest_class']}:{row['baseline_nearest_similarity']:.4f} "
            f"selected->{row['selected_nearest_class']}:{row['selected_nearest_similarity']:.4f} "
            f"delta_sep={row['separation_delta_selected_minus_baseline']:+.4f}"
        )


def analyze_candidate_pool(prompt_jsons, model, args, output_root, weights):
    class_names = list(PROMPT_LABEL_MAP.keys())
    all_payloads, candidate_texts, candidate_rows = load_candidate_pool(prompt_jsons, args.scoring_prompt_mode)
    class_anchor_texts = [PROMPT_LABEL_MAP[class_name] for class_name in class_names]

    print()
    print("#" * 110)
    print("Prompt JSON inputs:")
    for prompt_json in prompt_jsons:
        print("  -", Path(prompt_json).resolve())
    print("Classes:", len(class_names))
    print("Total candidate prompts:", len(candidate_texts))
    print("Class-name anchors:", ", ".join(class_anchor_texts))

    token_rows = get_clip_token_rows(candidate_texts, candidate_rows)
    token_fields = [
        "global_prompt_index",
        "source_file",
        "class_index",
        "class_name",
        "prompt_index",
        "clip_token_count_with_special",
        "clip_content_token_count",
        "raw_prompt_text",
        "encoded_prompt_text",
        "clip_token_ids_nonzero",
    ]
    write_csv(output_root / "all_candidate_clip_token_counts.csv", token_rows, token_fields)
    print()
    print("CLIP token counts for candidate prompts:")
    for row in token_rows:
        print(
            f"  global={row['global_prompt_index']:03d} "
            f"{row['source_stem']} {row['class_name']}#{row['prompt_index']} "
            f"tokens={row['clip_token_count_with_special']} "
            f"content={row['clip_content_token_count']} | {row['encoded_prompt_text']}"
        )

    all_summaries = []
    main_payload = None
    main_weighted_payload = None
    for space_name in args.spaces:
        space_dir = output_root / space_name
        space_dir.mkdir(parents=True, exist_ok=True)
        prompt_embeddings = normalize_np(encode_texts(model, candidate_texts, space_name, args.batch_size, args.device))
        anchor_embeddings = normalize_np(encode_texts(model, class_anchor_texts, space_name, args.batch_size, args.device))

        scored_rows, selected_rows, prompt_similarity, prompt_to_anchor, anchor_similarity, selected_similarity = rank_candidates(
            prompt_embeddings,
            anchor_embeddings,
            candidate_rows,
            class_names,
            weights,
            args.inter_mode,
        )
        weighted_rows = add_prompt_weights(scored_rows, args.prompt_weight_temperature)
        weighted_lookup = {
            row["global_prompt_index"]: {
                "prompt_weight": row["prompt_weight"],
                "weight_rank_in_class": row["weight_rank_in_class"],
                "prompt_weight_temperature": row["prompt_weight_temperature"],
            }
            for row in weighted_rows
        }
        for row in scored_rows:
            row.update(weighted_lookup[row["global_prompt_index"]])
        for row in selected_rows:
            row.update(weighted_lookup[row["global_prompt_index"]])
        baseline_rows = compare_baseline_and_selected(class_names, anchor_similarity, selected_similarity)
        prompt_to_anchor_rows = build_prompt_to_anchor_rows(scored_rows, prompt_to_anchor, class_names)

        score_fields = [
            "class_index",
            "class_name",
            "rank_in_class",
            "weight_rank_in_class",
            "source_file",
            "source_stem",
            "source_prompts_per_class",
            "prompt_index",
            "selection_score",
            "prompt_weight",
            "anchor_similarity",
            "intra_prompt_similarity",
            "inter_prompt_similarity",
            "anchor_margin",
            "best_other_anchor_class",
            "best_other_anchor_similarity",
            "raw_prompt_text",
            "encoded_prompt_text",
        ]
        write_csv(space_dir / "prompt_selection_scores_all_candidates.csv", scored_rows, score_fields)
        write_csv(space_dir / "selected_representative_prompts.csv", selected_rows, score_fields)
        prompt_to_anchor_fields = [
            "global_prompt_index",
            "source_file",
            "source_stem",
            "class_index",
            "class_name",
            "prompt_index",
            "own_class_similarity",
            "best_other_anchor_class",
            "best_other_anchor_similarity",
            "anchor_margin",
            *[f"sim_to_{class_name}" for class_name in class_names],
            "raw_prompt_text",
            "encoded_prompt_text",
        ]
        write_csv(space_dir / "prompt_to_class_anchor_similarity.csv", prompt_to_anchor_rows, prompt_to_anchor_fields)
        write_csv(
            space_dir / "baseline_classname_vs_selected_prompt_separation.csv",
            baseline_rows,
            [
                "class_name",
                "baseline_nearest_class",
                "baseline_nearest_similarity",
                "baseline_separation",
                "selected_nearest_class",
                "selected_nearest_similarity",
                "selected_separation",
                "separation_delta_selected_minus_baseline",
            ],
        )

        selected_payload = build_selected_payload(
            prompt_jsons,
            selected_rows,
            weights,
            args.inter_mode,
            space_name,
            args.scoring_prompt_mode,
        )
        save_json(space_dir / "selected_representative_prompts.json", selected_payload)
        if space_name == args.main_space:
            main_payload = selected_payload
            for source_file in sorted({row["source_file"] for row in weighted_rows}):
                weighted_payload = build_weighted_payload_for_source(
                    source_file,
                    weighted_rows,
                    prompt_jsons,
                    weights,
                    args.inter_mode,
                    space_name,
                    args.scoring_prompt_mode,
                    args.prompt_weight_temperature,
                )
                save_json(space_dir / f"weighted_prompt_bank_{Path(source_file).stem}.json", weighted_payload)
            first_source = Path(prompt_jsons[0]).name
            main_weighted_payload = build_weighted_payload_for_source(
                first_source,
                weighted_rows,
                prompt_jsons,
                weights,
                args.inter_mode,
                space_name,
                args.scoring_prompt_mode,
                args.prompt_weight_temperature,
            )

        summary = {
            "embedding_space": space_name,
            "prompt_jsons": [str(Path(path).resolve()) for path in prompt_jsons],
            "source_metadata": all_payloads,
            "classes": len(class_names),
            "total_candidate_prompts": len(candidate_texts),
            "weights": weights,
            "inter_mode": args.inter_mode,
            "selected_prompts": selected_rows,
            "baseline_vs_selected_separation": baseline_rows,
            "wrong_nearest_class_anchor_count": len([row for row in scored_rows if row["anchor_margin"] < 0]),
            "lowest_own_class_similarity_prompts": sorted(scored_rows, key=lambda row: row["anchor_similarity"])[:30],
            "lowest_total_score_prompts": sorted(scored_rows, key=lambda row: row["selection_score"])[:30],
        }
        save_json(space_dir / "prompt_selection_summary.json", summary)
        all_summaries.append(summary)
        print_console_summary(space_name, selected_rows, scored_rows, baseline_rows)

    if main_payload is not None:
        save_json(args.selected_output_json, main_payload)
        if main_weighted_payload is not None:
            save_json(args.weighted_output_json, main_weighted_payload)
        print()
        print("Main selected prompt bank for training:", args.selected_output_json)
        if main_weighted_payload is not None:
            print("Main weighted prompt bank for training:", args.weighted_output_json)

    save_json(output_root / "all_prompt_selection_summaries.json", all_summaries)
    return all_summaries


def main():
    args = parse_args()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    prompt_jsons = [Path(path).resolve() for path in args.prompt_jsons] if args.prompt_jsons else discover_prompt_jsons()
    if not prompt_jsons:
        raise FileNotFoundError("No prompt JSONs found. Expected code/ucf_manual_class_prompts_5x*.json or 10x*.json")
    for prompt_json in prompt_jsons:
        if not prompt_json.exists():
            raise FileNotFoundError(prompt_json)

    weights = {"anchor": args.w_anchor, "intra": args.w_intra, "inter": args.w_inter}
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DEVICE:", args.device)
    print("MODEL_PATH:", args.model_path)
    print("OUTPUT_DIR:", output_root)
    print("SELECTED_OUTPUT_JSON:", args.selected_output_json)
    print("WEIGHTED_OUTPUT_JSON:", args.weighted_output_json)
    print(
        "WEIGHTS:",
        weights,
        "| inter_mode:", args.inter_mode,
        "| main_space:", args.main_space,
        "| scoring_prompt_mode:", args.scoring_prompt_mode,
        "| prompt_weight_temperature:", args.prompt_weight_temperature,
    )

    model = instantiate_model(args.model_path, args.device)
    analyze_candidate_pool(prompt_jsons, model, args, output_root, weights)
    print()
    print("Done. Main output dir:", output_root)


if __name__ == "__main__":
    main()
