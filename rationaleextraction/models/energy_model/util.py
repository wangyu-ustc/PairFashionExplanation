import os
import argparse
import re
from collections import namedtuple
import numpy as np
import torch
import random
import math
import collections


from models.energy_model.constants import UNK_TOKEN, PAD_TOKEN
from torch.nn.init import _calculate_fan_in_and_fan_out
from torch import nn
from nltk.tokenize import word_tokenize
from tqdm import tqdm

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


def filereader(path, label_field="label"):
    """read jigsaw lines"""
    # with open(path, mode="r", encoding="utf-8") as f:
    #     for line in f:
    #         yield line.strip().replace("\\", "")
    x=[]
    y=[]
    with open(path) as f:
        for i, line in enumerate(tqdm(f)):
                try:
                    d = eval(line)
                    
                    x.append(d["document"])
                    y.append(int(d[label_field]))
                except:
                    print("Error evaluating / tokenizing"
                          " line {}, skipping it".format(i))
                    pass
    print("label field: ", label_field)
    frequency = collections.Counter(y)
    print(frequency)
    return x, y, frequency

def advfilereader(path):
    """read jigsaw lines"""
    # with open(path, mode="r", encoding="utf-8") as f:
    #     for line in f:
    #         yield line.strip().replace("\\", "")
    x=[]
    y=[]
    adv_y=[]
    with open(path) as f:
        for i, line in enumerate(tqdm(f)):
                try:
                    d = eval(line)
                    
                    x.append(d["document"])
                    y.append(int(d["label"]))
                    adv_y.append(int(d["gender"]))
                except:
                    print("Error evaluating / tokenizing"
                          " line {}, skipping it".format(i))
                    pass
    toxic_frequency = collections.Counter(y)
    print("toxic label frequency", toxic_frequency)
    gender_frequency =  collections.Counter(adv_y)
    print("toxic label frequency", gender_frequency)
    return x,y, adv_y, toxic_frequency, gender_frequency

def tokens_from_treestring(s):
    """extract the tokens from a sentiment tree"""
    return re.findall(r"\([0-9] ([^\(\)]+)\)", s)


def token_labels_from_treestring(s):
    """extract token labels from sentiment tree"""
    return list(map(int, re.findall(r"\(([0-9]) [^\(\)]", s)))


Example = namedtuple("Example", ["tokens", "label", "token_labels"])
AdvExample = namedtuple("AdvExample", ["tokens", "label", "token_labels", "adv_label"])


def jigsaw_reader(path, lower=False, label_field="label"):
    """
    Reads in examples
    :param path:
    :param lower:
    :return:
    """
    lines, labels, _=filereader(path, label_field=label_field)
    for line, label in zip(lines, labels):
        line = line.lower() if lower else line
        # line = re.sub("\\\\", "", line)  # fix escape
        
        tokens=word_tokenize(line)
        token_labels = [0]*len(tokens)
        # label = int(line[1])
        yield Example(tokens=tokens, label=label, token_labels=token_labels)
    print("Finishing load ", path)

def jigsaw_reader_adv(path, lower=False):
    """
    Reads in examples
    :param path:
    :param lower:
    :return:
    """
    lines, labels, adv_labels, _, _=advfilereader(path)
    for idx in range(0,len(lines)):
        line=lines[idx]
        line = line.lower() if lower else line
        # line = re.sub("\\\\", "", line)  # fix escape

        tokens=word_tokenize(line)
        token_labels = [0]*len(tokens)
        
        label=labels[idx]
        adv_label=adv_labels[idx]
        # label = int(line[1])
        yield AdvExample(tokens=tokens, label=label, token_labels=token_labels, adv_label=adv_label)
    print("Finishing load ", path)

def print_parameters(model):
    """Prints model parameters"""
    total = 0
    for name, p in model.named_parameters():
        total += np.prod(p.shape)
        print("{:24s} {:12s} requires_grad={}".format(name, str(list(p.shape)),
                                                      p.requires_grad))
    print("\nTotal parameters: {}\n".format(total))


def load_glove(glove_path, vocab, glove_dim=300):
    """
    Load Glove embeddings and update vocab.
    :param glove_path:
    :param vocab:
    :param glove_dim:
    :return:
    """
    vectors = []
    w2i = {}
    i2w = []

    # Random embedding vector for unknown words
    vectors.append(np.random.uniform(
        -0.05, 0.05, glove_dim).astype(np.float32))
    w2i[UNK_TOKEN] = 0
    i2w.append(UNK_TOKEN)

    # Zero vector for padding
    vectors.append(np.zeros(glove_dim).astype(np.float32))
    w2i[PAD_TOKEN] = 1
    i2w.append(PAD_TOKEN)

    with open(glove_path, mode="r", encoding="utf-8") as f:
        for line in f:
            word, vec = line.split(u' ', 1)
            w2i[word] = len(vectors)
            i2w.append(word)
            vectors.append(np.array(vec.split(), dtype=np.float32))

    # fix brackets
    w2i[u'-LRB-'] = w2i.pop(u'(')
    w2i[u'-RRB-'] = w2i.pop(u')')

    i2w[w2i[u'-LRB-']] = u'-LRB-'
    i2w[w2i[u'-RRB-']] = u'-RRB-'

    vocab.w2i = w2i
    vocab.i2w = i2w

    return np.stack(vectors)


def get_minibatch(data, batch_size=25, shuffle=False):
    """Return minibatches, optional shuffling"""

    if shuffle:
        print("Shuffling training data")
        random.shuffle(data)  # shuffle training data each epoch

    batch = []

    # yield minibatches
    for example in data:
        batch.append(example)

        if len(batch) == batch_size:
            yield batch
            batch = []

    # in case there is something left
    if len(batch) > 0:
        yield batch


def pad(tokens, length, pad_value=1):
    """add padding 1s to a sequence to that it has the desired length"""
    return tokens + [pad_value] * (length - len(tokens))


def prepare_minibatch(x, targets, device, return_reverse_map=False, concat=False, return_idx=False):

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

        x1, x2, mask1, mask2, lengths1, lengths2, targets = x1[idxes], x2[idxes], mask1[idxes], mask2[idxes], lengths1[idxes], lengths2[idxes], targets[idxes]

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
            if return_idx:
                return idxes, x1.to(device), x2.to(device), mask1.to(device), mask2.to(device), targets, reverse_map1, reverse_map2
            else:
                return x1.to(device), x2.to(device), mask1.to(device), mask2.to(device), targets, reverse_map1, reverse_map2
        else:
            if return_idx:
                return idxes, x1.to(device), x2.to(device), mask1.to(device), mask2.to(device), targets
            else:
                return x1.to(device), x2.to(device), mask1.to(device), mask2.to(device), targets
    
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
        targets = targets[idxes]
        assert len(lengths.shape) == 1
        x = x[:, :torch.max(lengths)]
        mask = mask[:, :torch.max(lengths)]
        # x[torch.where(x == 50257)] = 0
        if return_reverse_map:
            if return_idx:
                return idxes, x.to(device), mask.to(device), targets.to(device), reverse_map
            else:
                return x.to(device), mask.to(device), targets.to(device), reverse_map
        else:
            if return_idx:
                return idxes, x.to(device), mask.to(device), targets.to(device)
            else:
                return x.to(device), mask.to(device), targets.to(device)
  


def prepare_minibatch_adv(mb, vocab, device=None, sort=True):
    """
    Minibatch is a list of examples.
    This function converts words to IDs and returns
    torch tensors to be used as input/targets.
    """
    # batch_size = len(mb)
    reverse_map = None
    lengths = np.array([len(ex.tokens) for ex in mb])
    maxlen = lengths.max()

    # vocab returns 0 if the word is not there
    x = [pad([vocab.w2i.get(t, 0) for t in ex.tokens], maxlen) for ex in mb]
    y = [ex.label for ex in mb]
    adv_y=[ex.adv_label for ex in mb]

    x = np.array(x)
    y = np.array(y)
    adv_y = np.array(adv_y)

    if sort:  # required for LSTM
        sort_idx = np.argsort(lengths)[::-1]
        x = x[sort_idx]
        y = y[sort_idx]
        adv_y = adv_y[sort_idx]

        # to put back into the original order
        reverse_map = np.zeros(len(lengths), dtype=np.int32)
        for i, j in enumerate(sort_idx):
            reverse_map[j] = i

    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)
    adv_y = torch.from_numpy(adv_y).to(device)

    return x, y, adv_y, reverse_map

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


def get_args():
    parser = argparse.ArgumentParser(description='jigsaw')
    parser.add_argument('--save_path', type=str, default='jigsaw_results/energy/debug')
    parser.add_argument('--bias_model_path', type=str, default='jigsaw_results/gender/gender_rl_sparsity0003_coherence1/ model.pt')
    parser.add_argument('--adv_model_path', type=str, default='/data2/zexue/interpretable_predictions/latent_rationale/gender/jigsaw_results/gender/adv_latent_50pct_lmd001/model.pt')
    parser.add_argument('--resume_snapshot', type=str, default='')
    parser.add_argument('--label', type=str, default="label")

    parser.add_argument('--num_iterations', type=int, default=-30)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--eval_batch_size', type=int, default=512)
    parser.add_argument('--embed_size', type=int, default=300)
    parser.add_argument('--proj_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=150)

    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--threshold', type=float, default=1e-4)
    parser.add_argument('--cooldown', type=int, default=5)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=5.)

    parser.add_argument('--model',
                        choices=["baseline", "rl", "attention",
                                 "latent"],
                        default="baseline")
    parser.add_argument('--dist', choices=["", "hardkuma"],
                        default="")

    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=-1)
    parser.add_argument('--save_every', type=int, default=-1)

    parser.add_argument('--dropout', type=float, default=0.5)

    parser.add_argument('--layer', choices=["lstm"], default="lstm")
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb')

    parser.add_argument('--dependent-z', action='store_true',
                        help="make dependent decisions for z")

    # rationale settings for RL model
    parser.add_argument('--sparsity', type=float, default=0.0)
    parser.add_argument('--coherence', type=float, default=0.0)

    # rationale settings for HardKuma model
    parser.add_argument('--selection', type=float, default=0.3,
                        help="Target text selection rate for Lagrange.")
    parser.add_argument('--lasso', type=float, default=0.0)

    # lagrange settings
    parser.add_argument('--lagrange_lr', type=float, default=0.01,
                        help="learning rate for lagrange")
    parser.add_argument('--lagrange_alpha', type=float, default=0.99,
                        help="alpha for computing the running average")
    parser.add_argument('--lambda0_init', type=float, default=1e-4,
                        help="initial value for lambda0")
    parser.add_argument('--lambda1_init', type=float, default=1e-4,
                        help="initial value for lambda1")

    # misc
    parser.add_argument('--word_vectors', type=str,
                        default='/data2/zexue/interpretable_predictions/data/gender/glove.840B.300d.sst.txt')
    args = parser.parse_args()
    return args
