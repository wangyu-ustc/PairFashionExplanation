import os
import argparse
import pdb
import re
from collections import namedtuple
import numpy as np
import torch
import random
import math

from torch.nn.init import _calculate_fan_in_and_fan_out
from torch import nn
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.metrics import f1_score

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def find_ckpt_in_directory(path):
    for f in os.listdir(os.path.join(path, "")):
        if f.startswith('model'):
            return os.path.join(path, f)
    print("Could not find ckpt in {}".format(path))

def tokens_from_treestring(s):
    """extract the tokens from a sentiment tree"""
    return re.findall(r"\([0-9] ([^\(\)]+)\)", s)


def token_labels_from_treestring(s):
    """extract token labels from sentiment tree"""
    return list(map(int, re.findall(r"\(([0-9]) [^\(\)]", s)))


Example = namedtuple("Example", ["tokens", "label", "token_labels"])
AdvExample = namedtuple("AdvExample", ["tokens", "label", "token_labels", "adv_label"])

def print_parameters(model):
    """Prints model parameters"""
    total = 0
    for name, p in model.named_parameters():
        total += np.prod(p.shape)
        print("{:24s} {:12s} requires_grad={}".format(name, str(list(p.shape)),
                                                      p.requires_grad))
    print("\nTotal parameters: {}\n".format(total))


def prepare_minibatch(x, targets, device, sort_by_length=True, return_reverse_map=False, concat=False):

    targets = targets.to(device)

    if not concat: 
        x1, x2 = x
        mask1 = x1 != 50257
        mask2 = x2 != 50257
        mask1 = mask1.byte()
        mask2 = mask2.byte()

        lengths1 = mask1.sum(1)
        lengths2 = mask2.sum(1)

        idxes = torch.where((lengths1 > 0) & (lengths2 > 0))

        if len(idxes[0]) == 0:
            return None

        x1, x2, mask1, mask2, lengths1, lengths2, targets = x1[idxes], x2[idxes], mask1[idxes], mask2[idxes], lengths1[idxes], lengths2[idxes], targets[idxes]


        if sort_by_length:
            idxes = torch.argsort(lengths1, descending=True)
            reverse_map1 = torch.argsort(idxes)

            if len(idxes) == 0: return None

            x1 = x1[idxes]
            mask1 = mask1[idxes]
            assert len(lengths1.shape) == 1
            
            x1 = x1[:, :torch.max(lengths1)]
            mask1 = mask1[:, :torch.max(lengths1)]

            idxes = torch.argsort(lengths2, descending=True)
            reverse_map2 = torch.argsort(idxes)
            x2 = x2[idxes]
            mask2 = mask2[idxes]
            x2 = x2[:, :torch.max(lengths2)]
            mask2 = mask2[:, :torch.max(lengths2)]

            if return_reverse_map:
                return x1.to(device), x2.to(device), mask1.to(device), mask2.to(device), targets, reverse_map1, reverse_map2
            else:
                return x1.to(device), x2.to(device), mask1.to(device), mask2.to(device), targets
        else:
            x1 = x1[:, :torch.max(lengths1)]
            mask1 = mask1[:, :torch.max(lengths1)]
            x2 = x2[:, :torch.max(lengths2)]
            mask2 = mask2[:, :torch.max(lengths2)]

            return idxes, x1.to(device), x2.to(device), mask1.to(device), mask2.to(device), targets
    
    else:
        mask = x != 50257
        mask = mask.byte()
        lengths = mask.sum(1)
        idxes = torch.where(lengths > 0)
        x, mask, targets = x[idxes], mask[idxes], targets[idxes]
        idxes = torch.argsort(lengths, descending=True)
        reverse_map = torch.argsort(idxes)
        x = x[idxes]
        mask = mask[idxes]
        assert len(lengths.shape) == 1
        x = x[:, :torch.max(lengths)]
        mask = mask[:, :torch.max(lengths)]
        if return_reverse_map:
            return x.to(device), mask.to(device), targets.to(device), reverse_map
        else:
            return x.to(device), mask.to(device), targets.to(device)
        



def prepare_minibatch_adv(mb, tokenizer, device=None, sort=True):
    """
    Minibatch is a list of examples.
    This function converts words to IDs and returns
    torch tensors to be used as input/targets.
    """
    # batch_size = len(mb)

    prompts, prompts_attention_masks = tokenizer([ex.tokens for ex in mb], return_tensors='pt', padding=True).values()
    labels, labels_attention_masks = tokenizer([ex.label for ex in mb], return_tensors='pt', padding=True).values()

    reverse_map = None
    lengths = prompts_attention_masks.sum(dim=1)
    maxlen = lengths.max()

    # vocab returns 0 if the word is not there
    # x = [pad([vocab.w2i.get(t, 0) for t in ex.tokens], maxlen) for ex in mb]
    # y = [ex.label for ex in mb]
    adv_y = [ex.adv_label for ex in mb]

    # x = np.array(x)
    # y = np.array(y)
    adv_y = np.array(adv_y)

    if sort:  # required for LSTM
        sort_idx = torch.argsort(lengths, descending=True)
        prompts = prompts[sort_idx]
        prompts_attention_masks = prompts_attention_masks[sort_idx]
        labels = labels[sort_idx]
        labels_attention_masks = labels_attention_masks[sort_idx]
        adv_y = adv_y[sort_idx]

        # to put back into the original order
        reverse_map = np.zeros(len(lengths), dtype=np.int32)
        for i, j in enumerate(sort_idx):
            reverse_map[j] = i

    # x = torch.from_numpy(x).to(device)
    # y = torch.from_numpy(y).to(device)
    if len(mb) > 1:
        adv_y = torch.from_numpy(adv_y).to(device)
    else:
        adv_y = torch.tensor([adv_y]).to(device)
    prompts, prompts_attention_masks = prompts.to(device), prompts_attention_masks.to(device)
    labels, labels_attention_masks = labels.to(device), labels_attention_masks.to(device)

    # return x, y, adv_y, reverse_map
    return prompts, prompts_attention_masks, labels, labels_attention_masks, adv_y, reverse_map

def plot_dataset(model, data, batch_size=100, device=None, save_path=".",
                 ext="pdf"):
    """Accuracy of a model on given data set (using minibatches)"""

    model.eval()  # disable dropout
    sent_id = 0

    for mb in get_minibatch(data, batch_size=batch_size, shuffle=False):
        x, targets, reverse_map = prepare_minibatch(
            mb, model.vocab, device=device, sort=True)

        with torch.no_grad():
            logits = model(x)

            alphas = model.alphas if hasattr(model, "alphas") else None
            z = model.z if hasattr(model, "z") else None

        # reverse sort
        alphas = alphas[reverse_map] if alphas is not None else None
        z = z.squeeze(1).squeeze(-1)  # make [B, T]
        z = z[reverse_map] if z is not None else None

        for i, ex in enumerate(mb):
            tokens = ex.tokens

            if alphas is not None:
                alpha = alphas[i][:len(tokens)]
                alpha = alpha[None, :]
                path = os.path.join(
                    save_path, "plot{:04d}.alphas.{}".format(sent_id, ext))
                plot_heatmap(alpha, column_labels=tokens, output_path=path)

            # print(tokens)
            # print(" ".join(["%4.2f" % x for x in alpha]))

            # z is [batch_size, num_samples, time]
            if z is not None:

                zi = z[i, :len(tokens)]
                zi = zi[None, :]
                path = os.path.join(
                    save_path, "plot{:04d}.z.{}".format(sent_id, ext))
                plot_heatmap(zi, column_labels=tokens, output_path=path)

            sent_id += 1


def xavier_uniform_n_(w, gain=1., n=4):
    """
    Xavier initializer for parameters that combine multiple matrices in one
    parameter for efficiency. This is e.g. used for GRU and LSTM parameters,
    where e.g. all gates are computed at the same time by 1 big matrix.
    :param w:
    :param gain:
    :param n:
    :return:
    """
    with torch.no_grad():
        fan_in, fan_out = _calculate_fan_in_and_fan_out(w)
        assert fan_out % n == 0, "fan_out should be divisible by n"
        fan_out = fan_out // n
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        nn.init.uniform_(w, -a, a)


def initialize_model_(model):
    """
    Model initialization.

    :param model:
    :return:
    """
    # Custom initialization
    print("Glorot init")
    for name, p in model.named_parameters():
        if name.startswith("embed") or "lagrange" in name:
            print("{:10s} {:20s} {}".format("unchanged", name, p.shape))
        elif "lstm" in name and len(p.shape) > 1:
            print("{:10s} {:20s} {}".format("xavier_n", name, p.shape))
            xavier_uniform_n_(p)
        elif len(p.shape) > 1:
            print("{:10s} {:20s} {}".format("xavier", name, p.shape))
            torch.nn.init.xavier_uniform_(p)
        elif "bias" in name:
            print("{:10s} {:20s} {}".format("zeros", name, p.shape))
            torch.nn.init.constant_(p, 0.)
        else:
            print("{:10s} {:20s} {}".format("unchanged", name, p.shape))


def get_predict_args():
    parser = argparse.ArgumentParser(description='jigsaw prediction')
    parser.add_argument('--ckpt', type=str, default="path_to_checkpoint",
                        required=True)
    parser.add_argument('--plot', action="store_true", default=False)
    args = parser.parse_args()
    return args

def train_valid_test_split(idxes, train_size, valid_size, test_size):

    train_idxes, valid_test_idxes = train_test_split(idxes, train_size=train_size)
    valid_idxes, test_idxes = train_test_split(valid_test_idxes, train_size=valid_size/(valid_size + test_size))

    return train_idxes, valid_idxes, test_idxes


def compute_acc(GroundTruth, predictedIndices, topN):
    precision = []
    recall = []
    NDCG = []
    MRR = []

    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        for i in range(len(predictedIndices)):  # for a user,
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0 / math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0 / (j + 1.0))
                            mrrFlag = False
                        userHit += 1

                    if idcgCount > 0:
                        idcg += 1.0 / math.log2(j + 2)
                        idcgCount = idcgCount - 1

                if (idcg != 0):
                    ndcg += (dcg / idcg)

                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])
                sumForNdcg += ndcg
                sumForMRR += userMRR

        precision.append(sumForPrecision / len(predictedIndices))
        recall.append(sumForRecall / len(predictedIndices))
        NDCG.append(sumForNdcg / len(predictedIndices))
        MRR.append(sumForMRR / len(predictedIndices))

    return precision, recall, NDCG, MRR

def evaluate(model, data_loader, tokenizer, batch_size=25, device=None, concat=False):
    

    """Accuracy of a model on given data set (using minibatches)"""

    model.eval()  # disable dropout

    # z statistics
    totals = defaultdict(float)
    z_totals = defaultdict(float)
    # histogram_totals = np.zeros(5).astype(np.int64)
    # z_histogram_totals = np.zeros(5).astype(np.int64)

    for batch in data_loader:

        x1, x2, targets = batch

        targets = targets.to(device)
        batch_size = targets.size(0)
        with torch.no_grad():
            x1, x2, mask1, mask2, targets, reverse_map1, reverse_map2 = prepare_minibatch([x1, x2], targets, device, return_reverse_map=True)
            logits, attn = model([x1, x2], [mask1, mask2], [reverse_map1, reverse_map2], return_attn=True)
            predictions = model.predict(logits)
            # loss, loss_optional = model.get_loss(logits, targets, [z1, z2], [mask1.bool(), mask2.bool()])
            loss = model.get_loss(logits, targets)

            if isinstance(loss, dict):
                loss = loss["main"]

            totals['loss'] += loss.item() * batch_size
        
        # add the number of correct predictions to the total correct
        totals['acc'] += (predictions == targets.view(-1)).sum().item()
        totals['f1'] += f1_score(targets.view(-1).cpu(), predictions.view(-1).cpu(), average='micro') * batch_size
        totals['total'] += batch_size

    result = {}

    # loss, accuracy, optional items
    totals['total'] += 1e-9
    for k, v in totals.items():
        if k != "total":
            result[k] = v / totals["total"]

    # z scores
    z_totals['total'] += 1e-9
    for k, v in z_totals.items():
        if k != "total":
            result[k] = v / z_totals["total"]

    if "p0" in result:
        result["selected"] = 1 - result["p0"]

    return result