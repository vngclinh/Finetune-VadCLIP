# ADDED: must be set before torch initialises cuBLAS, otherwise --deterministic cannot make
# matmul reductions reproducible. Harmless (and near-free) when --deterministic is off.
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random

from model import CLIPVAD
from ucf_test import test
from utils.dataset import UCFDataset
from utils.tools import get_prompt_text, get_batch_label
import ucf_option

def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat([instance_logits, tmp], dim=0)

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss

def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    model.to(device)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)
    ap_best = 0
    epoch = 0

    if args.use_checkpoint == True:
        checkpoint = torch.load(args.checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        print("checkpoint info:")
        print("epoch:", epoch+1, " ap:", ap_best)

    for e in range(args.max_epoch):
        model.train()
        loss_total1 = 0
        loss_total2 = 0
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        for i in range(min(len(normal_loader), len(anomaly_loader))):
            step = 0
            normal_features, normal_label, normal_lengths = next(normal_iter)
            anomaly_features, anomaly_label, anomaly_lengths = next(anomaly_iter)

            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)
            text_labels = list(normal_label) + list(anomaly_label)
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)

            text_features, logits1, logits2 = model(visual_features, None, prompt_text, feat_lengths) 
            #loss1
            loss1 = CLAS2(logits1, text_labels, feat_lengths, device) 
            loss_total1 += loss1.item()
            #loss2
            loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            loss_total2 += loss2.item()
            #loss3
            loss3 = torch.zeros(1).to(device)
            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            loss3 = loss3 / 13 * 1e-1

            loss = loss1 + loss2 + loss3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += i * normal_loader.batch_size * 2
            if step % 1280 == 0 and step != 0:
                print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total1 / (i+1), '| loss2: ', loss_total2 / (i+1), '| loss3: ', loss3.item())
                AUC, AP = test(model, testloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
                AP = AUC

                if AP > ap_best:
                    ap_best = AP 
                    checkpoint = {
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ap': ap_best}
                    torch.save(checkpoint, args.checkpoint_path)
                
        scheduler.step()
        
        torch.save(model.state_dict(), args.save_cur_path)
        # ADDED: archive the end-of-epoch weights so a dropped Colab session is recoverable.
        # Upstream only keeps model_cur.pth, overwritten every epoch. Off unless
        # --epoch-checkpoint-dir is given; torch.save consumes no RNG, so the training
        # trajectory is bit-identical either way.
        if args.epoch_checkpoint_dir:
            epoch_path = os.path.join(args.epoch_checkpoint_dir, 'model_epoch_%03d.pth' % (e + 1))
            torch.save(model.state_dict(), epoch_path)
            print('saved epoch checkpoint:', epoch_path)
        print('=== end of epoch', e + 1, '| best AUC so far:', ap_best, '===', flush=True)
        checkpoint = torch.load(args.checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

    checkpoint = torch.load(args.checkpoint_path, weights_only=False)
    torch.save(checkpoint['model_state_dict'], args.model_path)
    print('saved final model:', args.model_path)

def setup_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # ADDED: upstream leaves the line below commented out, so two runs of the SAME command with
    # the SAME seed still differ - GPU reductions are free to change order between runs.
    # --deterministic turns that off, making a run reproducible on the same GPU.
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: if any op has no deterministic kernel, warn instead of crashing the run.
        torch.use_deterministic_algorithms(True, warn_only=True)
        print('Deterministic mode ON (cudnn.deterministic, no benchmark, CUBLAS_WORKSPACE_CONFIG='
              + os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'unset') + ')')

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()
    setup_seed(args.seed, args.deterministic)

    # ADDED: upstream assumes a model/ directory already exists in the working directory.
    for path in [args.model_path, args.checkpoint_path, args.save_cur_path]:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if args.epoch_checkpoint_dir:
        os.makedirs(args.epoch_checkpoint_dir, exist_ok=True)

    label_map = dict({'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', 'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion', 'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery', 'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 'Vandalism': 'vandalism'})

    # ADDED: the trailing args.feature_root, and loader_kwargs. Everything else about the
    # construction order is untouched, because that order is what fixes the RNG stream.
    # With the defaults (--num-workers 0, --pin-memory false) loader_kwargs is empty and
    # these four lines are exactly upstream's.
    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs = {'num_workers': args.num_workers, 'persistent_workers': True}
    if args.pin_memory and device == 'cuda':
        loader_kwargs['pin_memory'] = True
    print('DataLoader kwargs:', loader_kwargs if loader_kwargs else '(upstream defaults)')

    normal_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, True, args.feature_root)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    anomaly_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, False, args.feature_root)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kwargs)

    test_dataset = UCFDataset(args.visual_length, args.test_list, True, label_map, False, args.feature_root)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, **loader_kwargs)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device)

    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)