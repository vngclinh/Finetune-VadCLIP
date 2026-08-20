import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import ucf_option_class_semantic
from model import CLIPVAD
from ucf_test_description import test
from utils.dataset_description import UCFDescriptionDataset
from utils.tools import get_batch_label, get_prompt_text


def length_to_int(length, max_length):
    if hasattr(length, "item"):
        length = length.item()
    length = int(length)
    return max(1, min(length, max_length))


def topk_count(valid_length):
    return max(1, min(valid_length // 16 + 1, valid_length))


def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        valid_length = length_to_int(lengths[i], logits.shape[1])
        tmp, _ = torch.topk(logits[i, :valid_length], k=topk_count(valid_length), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss


def CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        valid_length = length_to_int(lengths[i], logits.shape[1])
        tmp, _ = torch.topk(logits[i, :valid_length], k=topk_count(valid_length), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat([instance_logits, tmp], dim=0)

    return F.binary_cross_entropy(instance_logits, labels)


def text_regularization_loss(text_features, device):
    loss = torch.zeros(1).to(device)
    text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
    for j in range(1, text_features.shape[0]):
        text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
        loss += torch.abs(text_feature_normal @ text_feature_abr)
    return loss / 13 * 1e-1


def get_autocast_context(device, use_amp):
    if device == "cuda" and use_amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def normalize_text_features(features):
    return F.normalize(features.float(), dim=-1)


def encode_texts_with_clip(model, texts, batch_size=128):
    from clip import clip

    embeddings = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            tokens = clip.tokenize(batch, truncate=True).to(model.device)
            word_embeddings = model.clipmodel.encode_token(tokens)
            text_features = model.clipmodel.encode_text(word_embeddings, tokens)
            embeddings.append(text_features.detach().float())
    if was_training:
        model.train()
    return torch.cat(embeddings, dim=0)


def precompute_class_embeddings(model, prompt_text):
    class_embeddings = model.encode_textprompt(prompt_text).detach().float()
    class_embeddings = normalize_text_features(class_embeddings)
    print("Class semantic embeddings:", class_embeddings.shape)
    return class_embeddings


def precompute_description_embeddings(model, datasets):
    description_by_video = {}
    for dataset in datasets:
        for row in dataset.rows:
            video_id = row["video_id"]
            description = dataset.description_map.get(video_id, "")
            if description:
                description_by_video[video_id] = description

    video_ids = sorted(description_by_video)
    descriptions = [description_by_video[video_id] for video_id in video_ids]
    description_embeddings = encode_texts_with_clip(model, descriptions)
    description_embeddings = normalize_text_features(description_embeddings)
    print("Description semantic embeddings:", description_embeddings.shape)
    return {video_id: embedding for video_id, embedding in zip(video_ids, description_embeddings)}


def build_semantic_target(
    description_embeddings,
    video_ids,
    labels_raw,
    label_map,
    prompt_text,
    class_embeddings,
    semantic_alpha,
    semantic_temperature,
    device,
):
    batch_description_embeddings = torch.stack([description_embeddings[video_id] for video_id in video_ids], dim=0).to(device)
    batch_description_embeddings = normalize_text_features(batch_description_embeddings)
    similarities = batch_description_embeddings @ class_embeddings.to(device).t()
    q_desc = F.softmax(similarities / semantic_temperature, dim=-1)

    one_hot = torch.zeros_like(q_desc)
    class_index = {label: index for index, label in enumerate(prompt_text)}
    for i, label in enumerate(labels_raw):
        mapped_label = label_map.get(label, label)
        if mapped_label not in class_index:
            raise KeyError(f"Label '{label}' was mapped to '{mapped_label}', but it is not in prompt_text.")
        one_hot[i, class_index[mapped_label]] = 1.0

    q_final = (1 - semantic_alpha) * one_hot + semantic_alpha * q_desc
    q_final = q_final / q_final.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return q_desc, q_final


def pool_video_logits(logits2, logits1, labels_raw, lengths, pooling, top_k_ratio):
    pooled = []
    anomaly_scores = torch.sigmoid(logits1).squeeze(-1)
    for i in range(logits2.shape[0]):
        valid_len = length_to_int(lengths[i], logits2.shape[1])
        valid_logits = logits2[i, :valid_len]

        if labels_raw[i] == "Normal" or pooling == "mean":
            pooled.append(valid_logits.mean(dim=0))
            continue

        valid_scores = anomaly_scores[i, :valid_len]
        if pooling == "weighted":
            weights = valid_scores.clamp_min(1e-6)
            weights = weights / weights.sum()
            pooled.append((valid_logits * weights.unsqueeze(-1)).sum(dim=0))
        elif pooling == "topk":
            k = max(1, min(valid_len, int(round(valid_len * top_k_ratio))))
            top_indices = torch.topk(valid_scores, k=k, largest=True).indices
            pooled.append(valid_logits[top_indices].mean(dim=0))
        else:
            raise ValueError(f"Unsupported video logit pooling: {pooling}")
    return torch.stack(pooled, dim=0)


def class_semantic_loss(logits2, logits1, labels_raw, lengths, q_final, pooling, top_k_ratio):
    video_class_logits = pool_video_logits(logits2, logits1, labels_raw, lengths, pooling, top_k_ratio)
    log_p_video = F.log_softmax(video_class_logits.float(), dim=-1)
    return F.kl_div(log_p_video, q_final.float(), reduction="batchmean")


def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    model.to(device)
    Path(args.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save_cur_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.epoch_checkpoint_dir).mkdir(parents=True, exist_ok=True)

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    if args.use_pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model_path, map_location=device))
        print("Loaded pretrained model:", args.pretrained_model_path)
    else:
        print("Skipped VadCLIP checkpoint load. Training from scratch for VadCLIP-specific layers.")

    for clip_param in model.clipmodel.parameters():
        clip_param.requires_grad = False

    prompt_text = get_prompt_text(label_map)
    class_embeddings = precompute_class_embeddings(model, prompt_text)
    if args.cache_semantic_targets:
        description_embeddings = precompute_description_embeddings(model, [normal_loader.dataset, anomaly_loader.dataset])
    else:
        description_embeddings = None

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    use_amp = bool(args.use_amp and device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_score = 0

    print(
        "Class semantic config:",
        "| lambda_sem:", args.lambda_sem,
        "| semantic_alpha:", args.semantic_alpha,
        "| semantic_temperature:", args.semantic_temperature,
        "| video_logit_pooling:", args.video_logit_pooling,
        "| top_k_ratio:", args.top_k_ratio,
    )

    for e in range(args.max_epoch):
        model.train()
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        loss_total1 = 0
        loss_total2 = 0
        loss_total_sem = 0

        for i in range(min(len(normal_loader), len(anomaly_loader))):
            normal_features, normal_label, normal_lengths, normal_video_ids, normal_descriptions = next(normal_iter)
            anomaly_features, anomaly_label, anomaly_lengths, anomaly_video_ids, anomaly_descriptions = next(anomaly_iter)

            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device, non_blocking=True)
            text_labels_raw = list(normal_label) + list(anomaly_label)
            descriptions = list(normal_descriptions) + list(anomaly_descriptions)
            video_ids = list(normal_video_ids) + list(anomaly_video_ids)
            feat_lengths_tensor = torch.cat([normal_lengths, anomaly_lengths], dim=0)
            feat_lengths = [length_to_int(length, args.visual_length) for length in feat_lengths_tensor]
            text_labels = get_batch_label(text_labels_raw, prompt_text, label_map).to(device, non_blocking=True)

            if description_embeddings is None:
                desc_embeddings_batch = encode_texts_with_clip(model, descriptions)
                description_embeddings_batch = {
                    video_id: embedding for video_id, embedding in zip(video_ids, desc_embeddings_batch)
                }
            else:
                missing = [video_id for video_id in video_ids if video_id not in description_embeddings]
                if missing:
                    raise KeyError(f"Missing semantic description embeddings for video_ids: {missing[:10]}")
                description_embeddings_batch = description_embeddings

            _, q_final = build_semantic_target(
                description_embeddings_batch,
                video_ids,
                text_labels_raw,
                label_map,
                prompt_text,
                class_embeddings,
                args.semantic_alpha,
                args.semantic_temperature,
                device,
            )

            with get_autocast_context(device, use_amp):
                text_features, logits1, logits2 = model(visual_features, None, prompt_text, feat_lengths)

            loss1 = CLAS2(logits1.float(), text_labels.float(), feat_lengths, device)
            loss2 = CLASM(logits2.float(), text_labels.float(), feat_lengths, device)
            loss3 = text_regularization_loss(text_features.float(), device)
            loss_sem = class_semantic_loss(
                logits2.float(),
                logits1.float(),
                text_labels_raw,
                feat_lengths,
                q_final,
                args.video_logit_pooling,
                args.top_k_ratio,
            )
            loss = loss1 + loss2 + loss3 + args.lambda_sem * loss_sem

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_total1 += loss1.item()
            loss_total2 += loss2.item()
            loss_total_sem += loss_sem.item()

            step = i * normal_loader.batch_size * 2
            if args.eval_steps > 0 and step % args.eval_steps == 0 and step != 0:
                print(
                    "epoch:", e + 1,
                    "| step:", step,
                    "| loss1:", loss_total1 / (i + 1),
                    "| loss2:", loss_total2 / (i + 1),
                    "| loss_sem:", loss_total_sem / (i + 1),
                )
                auc, ap = test(model, testloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
                if auc > best_score:
                    best_score = auc
                    torch.save(model.state_dict(), args.checkpoint_path)

        scheduler.step()
        num_steps = max(1, min(len(normal_loader), len(anomaly_loader)))
        print(
            "epoch:", e + 1,
            "| loss1:", loss_total1 / num_steps,
            "| loss2:", loss_total2 / num_steps,
            "| loss_sem:", loss_total_sem / num_steps,
        )
        torch.save(model.state_dict(), args.save_cur_path)
        epoch_checkpoint_path = Path(args.epoch_checkpoint_dir) / f"model_epoch_{e + 1:03d}_class_semantic.pth"
        torch.save(model.state_dict(), epoch_checkpoint_path)
        print("Saved epoch checkpoint:", epoch_checkpoint_path)

    if best_score > 0:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    torch.save(model.state_dict(), args.output_model_path)
    print("Saved fine-tuned model:", args.output_model_path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option_class_semantic.parser.parse_args()
    setup_seed(args.seed)

    label_map = {
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

    pin_memory = bool(args.pin_memory and device == "cuda")
    dataloader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }
    if args.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True

    normal_dataset = UCFDescriptionDataset(
        args.visual_length, args.train_list, False, label_map, args.feature_root, args.description_json, True
    )
    normal_loader = DataLoader(
        normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **dataloader_kwargs
    )
    anomaly_dataset = UCFDescriptionDataset(
        args.visual_length, args.train_list, False, label_map, args.feature_root, args.description_json, False
    )
    anomaly_loader = DataLoader(
        anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **dataloader_kwargs
    )

    test_dataset = UCFDescriptionDataset(
        args.visual_length, args.test_list, True, label_map, args.feature_root, args.description_json, False,
        require_description=False
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, **dataloader_kwargs)

    model = CLIPVAD(
        args.classes_num,
        args.embed_dim,
        args.visual_length,
        args.visual_width,
        args.visual_head,
        args.visual_layers,
        args.attn_window,
        args.prompt_prefix,
        args.prompt_postfix,
        device,
    )

    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)
