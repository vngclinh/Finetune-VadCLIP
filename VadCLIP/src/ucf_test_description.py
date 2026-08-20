import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from utils.tools import get_batch_mask
from utils.ucf_detectionMAP import getDetectionMAP as dmAP


def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device):
    model.to(device)
    model.eval()

    element_logits2_stack = []

    with torch.no_grad():
        for i, item in enumerate(testdataloader):
            visual = item[0].squeeze(0)
            length = item[2]

            length = int(length)
            len_cur = length
            if len_cur < maxlen:
                visual = visual.unsqueeze(0)

            visual = visual.to(device)

            lengths = torch.zeros(int(length / maxlen) + 1)
            for j in range(int(length / maxlen) + 1):
                if j == 0 and length < maxlen:
                    lengths[j] = length
                elif j == 0 and length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                elif length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                else:
                    lengths[j] = length
            lengths = lengths.to(int)
            padding_mask = get_batch_mask(lengths, maxlen).to(device)
            _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
            prob2 = 1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1)
            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))

            if i == 0:
                ap1 = prob1
                ap2 = prob2
            else:
                ap1 = torch.cat([ap1, prob1], dim=0)
                ap2 = torch.cat([ap2, prob2], dim=0)

            element_logits2 = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy()
            element_logits2 = np.repeat(element_logits2, 16, 0)
            element_logits2_stack.append(element_logits2)

    ap1 = ap1.cpu().numpy().tolist()
    ap2 = ap2.cpu().numpy().tolist()
    score_len = len(np.repeat(ap1, 16))
    if score_len != len(gt):
        raise ValueError(
            "Test score length does not match ground truth length. "
            f"score_len={score_len}, gt_len={len(gt)}. "
            "Use the full 290-video UCF test list for gt_ucf.npy, not the UCA-filtered description test list."
        )

    roc1 = roc_auc_score(gt, np.repeat(ap1, 16))
    ap_score1 = average_precision_score(gt, np.repeat(ap1, 16))
    roc2 = roc_auc_score(gt, np.repeat(ap2, 16))
    ap_score2 = average_precision_score(gt, np.repeat(ap2, 16))

    print("AUC1: ", roc1, " AP1: ", ap_score1)
    print("AUC2: ", roc2, " AP2:", ap_score2)

    dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
    average_map = 0
    for i in range(5):
        print("mAP@{0:.1f} ={1:.2f}%".format(iou[i], dmap[i]))
        average_map += dmap[i]
    average_map = average_map / (i + 1)
    print("average MAP: {:.2f}".format(average_map))

    return roc1, ap_score1
