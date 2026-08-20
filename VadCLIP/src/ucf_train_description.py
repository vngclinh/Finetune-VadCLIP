import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import ucf_option_description
from model import CLIPVAD
from model_description import CLIPVADDescription
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

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss


def load_pretrained_state(model, model_path, device):
    state_dict = torch.load(model_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("Missing keys initialized randomly:", missing)
    if unexpected:
        print("Unexpected keys ignored:", unexpected)


def set_trainable_scope(model, trainable_scope):
    for param in model.parameters():
        param.requires_grad = False

    if trainable_scope == "all":
        for name, param in model.named_parameters():
            if not name.startswith("clipmodel."):
                param.requires_grad = True
    elif trainable_scope == "projection_only":
        for param in model.desc_projection.parameters():
            param.requires_grad = True
    elif trainable_scope == "projection_mlp_classifier":
        modules = [model.desc_projection, model.mlp1, model.mlp2, model.classifier]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True
    else:
        raise ValueError(f"Unsupported trainable scope: {trainable_scope}")

    for clip_param in model.clipmodel.parameters():
        clip_param.requires_grad = False


def build_teacher(args, device):
    if not args.use_pretrained_model:
        raise ValueError("Baseline distillation requires --use-pretrained-model true.")
    teacher = CLIPVAD(
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
    teacher.load_state_dict(torch.load(args.pretrained_model_path, map_location=device))
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def mean_pool_video_features(visual_features, lengths):
    pooled = []
    for i in range(visual_features.shape[0]):
        valid_len = int(lengths[i].item()) if hasattr(lengths[i], "item") else int(lengths[i])
        valid_len = max(1, min(valid_len, visual_features.shape[1]))
        pooled.append(visual_features[i, :valid_len].mean(dim=0))
    return torch.stack(pooled, dim=0)


def description_pool_video_features(visual_features, logits1, labels_raw, lengths, pooling, top_k_ratio):
    pooled = []
    anomaly_scores = torch.sigmoid(logits1).squeeze(-1)
    for i in range(visual_features.shape[0]):
        valid_len = length_to_int(lengths[i], visual_features.shape[1])
        valid_features = visual_features[i, :valid_len]
        if labels_raw[i] == "Normal" or pooling == "mean":
            pooled.append(valid_features.mean(dim=0))
            continue

        valid_scores = anomaly_scores[i, :valid_len]
        if pooling == "weighted":
            weights = valid_scores.clamp_min(1e-6)
            weights = weights / weights.sum()
            pooled.append((valid_features * weights.unsqueeze(-1)).sum(dim=0))
        elif pooling == "topk":
            k = max(1, min(valid_len, int(round(valid_len * top_k_ratio))))
            top_indices = torch.topk(valid_scores, k=k, largest=True).indices
            pooled.append(valid_features[top_indices].mean(dim=0))
        else:
            raise ValueError(f"Unsupported description pooling: {pooling}")
    return torch.stack(pooled, dim=0)


def unique_contrastive_indices(video_ids, labels_raw, contrastive_samples):
    seen = set()
    indices = []
    for index, (video_id, label) in enumerate(zip(video_ids, labels_raw)):
        if contrastive_samples == "abnormal" and label == "Normal":
            continue
        if video_id in seen:
            continue
        seen.add(video_id)
        indices.append(index)
    return indices


def description_alignment_loss(
    model,
    visual_features,
    logits1,
    labels_raw,
    descriptions,
    lengths,
    description_embeddings=None,
    pooling="topk",
    top_k_ratio=0.15,
    use_projection=True,
):
    video_embedding = description_pool_video_features(
        visual_features, logits1, labels_raw, lengths, pooling, top_k_ratio
    )
    if use_projection:
        video_embedding = model.desc_projection(video_embedding)
    if description_embeddings is None:
        description_embedding = model.encode_description(list(descriptions))
    else:
        description_embedding = description_embeddings
    video_embedding = F.normalize(video_embedding, dim=-1)
    description_embedding = F.normalize(description_embedding, dim=-1)
    return 1 - (video_embedding * description_embedding).sum(dim=-1).mean()


def description_contrastive_loss(
    model,
    visual_features,
    logits1,
    labels_raw,
    video_ids,
    lengths,
    description_embeddings,
    pooling="topk",
    top_k_ratio=0.15,
    use_projection=True,
    temperature=0.07,
    contrastive_samples="all",
):
    contrastive_indices = unique_contrastive_indices(video_ids, labels_raw, contrastive_samples)
    if len(contrastive_indices) < 2:
        return torch.zeros(1, device=visual_features.device).squeeze()

    video_embedding = description_pool_video_features(
        visual_features, logits1, labels_raw, lengths, pooling, top_k_ratio
    )
    if use_projection:
        video_embedding = model.desc_projection(video_embedding)

    index_tensor = torch.tensor(contrastive_indices, device=visual_features.device, dtype=torch.long)
    video_embedding = video_embedding.index_select(0, index_tensor)
    description_embedding = description_embeddings.index_select(0, index_tensor)

    video_embedding = F.normalize(video_embedding, dim=-1)
    description_embedding = F.normalize(description_embedding, dim=-1)
    logits = video_embedding @ description_embedding.t() / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_v2t = F.cross_entropy(logits, labels)
    loss_t2v = F.cross_entropy(logits.t(), labels)
    return (loss_v2t + loss_t2v) / 2


def distillation_loss(student_logits1, student_logits2, teacher_logits1, teacher_logits2):
    student_cls = torch.sigmoid(student_logits1.float())
    teacher_cls = torch.sigmoid(teacher_logits1.float()).detach()
    student_align = F.softmax(student_logits2.float(), dim=-1)
    teacher_align = F.softmax(teacher_logits2.float(), dim=-1).detach()
    return F.mse_loss(student_cls, teacher_cls) + F.mse_loss(student_align, teacher_align)


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


def precompute_description_embeddings(model, datasets, device, batch_size=128):
    description_by_video = {}
    for dataset in datasets:
        for row in dataset.rows:
            video_id = row["video_id"]
            description = dataset.description_map.get(video_id, "")
            if description:
                description_by_video[video_id] = description

    if not description_by_video:
        return {}

    model.eval()
    embedding_cache = {}
    video_ids = sorted(description_by_video)
    with torch.no_grad():
        for start in range(0, len(video_ids), batch_size):
            batch_ids = video_ids[start:start + batch_size]
            descriptions = [description_by_video[video_id] for video_id in batch_ids]
            embeddings = model.encode_description(descriptions).detach().to(device)
            for video_id, embedding in zip(batch_ids, embeddings):
                embedding_cache[video_id] = embedding

    print("Cached description embeddings:", len(embedding_cache))
    return embedding_cache


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
        load_pretrained_state(model, args.pretrained_model_path, device)
        print("Loaded pretrained model:", args.pretrained_model_path)
    else:
        print("Skipped VadCLIP checkpoint load. Training from scratch for VadCLIP-specific layers.")

    set_trainable_scope(model, args.trainable_scope)
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print("Trainable parameters:", trainable_params)

    teacher_model = build_teacher(args, device) if args.distill_baseline and args.lambda_distill > 0 else None
    if teacher_model is not None:
        print("Loaded frozen baseline teacher for distillation.")

    description_embedding_cache = None
    if args.cache_description_embeddings:
        description_embedding_cache = precompute_description_embeddings(
            model, [normal_loader.dataset, anomaly_loader.dataset], device
        )

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    use_amp = bool(args.use_amp and device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    prompt_text = get_prompt_text(label_map)
    best_score = 0
    print(
        "Description loss:",
        args.desc_loss_type,
        "| lambda_desc:", args.lambda_desc,
        "| lambda_contrastive:", args.lambda_contrastive,
        "| temperature:", args.contrastive_temperature,
        "| samples:", args.contrastive_samples,
        "| pooling:", args.desc_pooling,
        "| top_k_ratio:", args.top_k_ratio,
        "| lambda_distill:", args.lambda_distill,
    )

    for e in range(args.max_epoch):
        model.train()
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        loss_total1 = 0
        loss_total2 = 0
        loss_total_desc = 0
        loss_total_distill = 0

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

            description_embeddings = None
            if description_embedding_cache is not None:
                missing_cached = [video_id for video_id in video_ids if video_id not in description_embedding_cache]
                if missing_cached:
                    raise KeyError(f"Missing cached description embeddings for video_ids: {missing_cached[:10]}")
                description_embeddings = torch.stack([description_embedding_cache[video_id] for video_id in video_ids], dim=0)

            with get_autocast_context(device, use_amp):
                text_features, logits1, logits2, encoded_visual_features = model(
                    visual_features, None, prompt_text, feat_lengths, return_visual=True
                )
                if teacher_model is not None:
                    with torch.no_grad():
                        _, teacher_logits1, teacher_logits2 = teacher_model(
                            visual_features, None, prompt_text, feat_lengths
                        )
                else:
                    teacher_logits1 = None
                    teacher_logits2 = None

            # Keep losses in FP32. BCE after sigmoid is unsafe inside autocast.
            if description_embeddings is not None:
                description_embeddings = description_embeddings.float()
            loss1 = CLAS2(logits1.float(), text_labels.float(), feat_lengths, device)
            loss2 = CLASM(logits2.float(), text_labels.float(), feat_lengths, device)
            loss3 = text_regularization_loss(text_features.float(), device)
            if args.desc_loss_type == "contrastive":
                loss_desc = description_contrastive_loss(
                    model,
                    encoded_visual_features.float(),
                    logits1.float(),
                    text_labels_raw,
                    video_ids,
                    feat_lengths,
                    description_embeddings,
                    args.desc_pooling,
                    args.top_k_ratio,
                    args.use_desc_projection,
                    args.contrastive_temperature,
                    args.contrastive_samples,
                )
                desc_weight = args.lambda_contrastive
            else:
                loss_desc = description_alignment_loss(
                    model,
                    encoded_visual_features.float(),
                    logits1.float(),
                    text_labels_raw,
                    descriptions,
                    feat_lengths,
                    description_embeddings,
                    args.desc_pooling,
                    args.top_k_ratio,
                    args.use_desc_projection,
                )
                desc_weight = args.lambda_desc
            if teacher_model is not None:
                loss_distill = distillation_loss(logits1, logits2, teacher_logits1, teacher_logits2)
            else:
                loss_distill = torch.zeros(1, device=device)
            loss = loss1 + loss2 + loss3 + desc_weight * loss_desc + args.lambda_distill * loss_distill

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_total1 += loss1.item()
            loss_total2 += loss2.item()
            loss_total_desc += loss_desc.item()
            loss_total_distill += loss_distill.item()

            step = i * normal_loader.batch_size * 2
            if args.eval_steps > 0 and step % args.eval_steps == 0 and step != 0:
                print(
                    "epoch:", e + 1,
                    "| step:", step,
                    "| loss1:", loss_total1 / (i + 1),
                    "| loss2:", loss_total2 / (i + 1),
                    "| loss_desc:", loss_total_desc / (i + 1),
                    "| loss_distill:", loss_total_distill / (i + 1),
                )
                auc, ap = test(model, testloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
                score = auc
                if score > best_score:
                    best_score = score
                    torch.save(model.state_dict(), args.checkpoint_path)

        scheduler.step()
        print(
            "epoch:", e + 1,
            "| loss1:", loss_total1 / max(1, min(len(normal_loader), len(anomaly_loader))),
            "| loss2:", loss_total2 / max(1, min(len(normal_loader), len(anomaly_loader))),
            "| loss_desc:", loss_total_desc / max(1, min(len(normal_loader), len(anomaly_loader))),
            "| loss_distill:", loss_total_distill / max(1, min(len(normal_loader), len(anomaly_loader))),
        )
        torch.save(model.state_dict(), args.save_cur_path)
        epoch_checkpoint_path = Path(args.epoch_checkpoint_dir) / f"model_epoch_{e + 1:03d}_description.pth"
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
    args = ucf_option_description.parser.parse_args()
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

    model = CLIPVADDescription(
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
