import os
import pdb
import sys
import time
from tqdm import tqdm
import pandas as pd
from collections import OrderedDict
import json
import argparse

import math
import torch
import torch.optim
from torch.optim import Adam
import numpy as np
from models.common.util import make_kv_string
from models.energy_model.models.model_helpers import build_model
from models.energy_model.util import get_args, \
    prepare_minibatch, get_minibatch, load_glove, print_parameters, \
    initialize_model_, get_device
from transformers import BertTokenizer
from models.energy_model.dataset import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from nltk.stem.porter import *

device = get_device()
print("device:", device)


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


def train_valid_test_split(idxes, train_size, valid_size, test_size):
    train_idxes, valid_test_idxes = train_test_split(idxes, train_size=train_size)
    valid_idxes, test_idxes = train_test_split(valid_test_idxes, train_size=valid_size / (valid_size + test_size))

    return train_idxes, valid_idxes, test_idxes


def set_seed(seed):
    if seed == -1:
        seed = int(torch.randint(0, 2 ** 32 - 1, (1,)).item())

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_blocks(sentence, tokenizer):
    blocks = []
    last_length = 0
    for l in range(1, len(sentence) + 1):
        current_length = len(tokenizer(" ".join(sentence[:l]))['input_ids'][1:-1])
        blocks.append(current_length - last_length)
        last_length = current_length
    return blocks


def test_recall(cfg):
    cfg = vars(cfg)

    set_seed(cfg['seed'])

    for k, v in cfg.items():
        print("{} : {}".format(k, v))

    batch_size = cfg["batch_size"]

    gpt2_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    print("Loading data")
    dataset = Dataset(
        path=cfg['path'],
        tokenizer=gpt2_tokenizer,
        debug=cfg['debug'],
        concat=True
    )

    train_idxes, valid_idxes, test_idxes = train_valid_test_split(np.arange(len(dataset)), 0.8, 0.1, 0.1)
    train_idxes = np.concatenate([train_idxes, valid_idxes])

    train_idxes = train_idxes[np.where(train_idxes < len(dataset.data))]
    test_idxes = test_idxes[np.where(test_idxes < len(dataset.data))]

    train_positive_pairs = {
        (i, j) for (i, j) in dataset.data[train_idxes]
    }

    num_workers = 0 if cfg['num_workers'] == -1 else cfg['num_workers']

    model = build_model(cfg)
    initialize_model_(model)

    assert cfg['pretrained_weights'] is not None, "You should specify pretrained weights"

    model_ckpt = torch.load(cfg["pretrained_weights"], map_location=device)
    cfg = model_ckpt["cfg"]
    model = build_model(cfg)
    model.load_state_dict(model_ckpt["state_dict"])
    model.to(device)

    test_positive_pairs = dataset.data[test_idxes]

    def construct_interaction_dict(test_positive_pairs):
        interactions = {}

        def add_item_to_list(i, j):
            if i in interactions:
                interactions[i].append(j)
            else:
                interactions[i] = [j]

        for (i, j) in test_positive_pairs:
            add_item_to_list(i, j)
            add_item_to_list(j, i)

        for key in interactions:
            interactions[key] = list(set(interactions[key]))

        return interactions

    train_interactions = construct_interaction_dict(dataset.data[train_idxes])
    interactions = construct_interaction_dict(test_positive_pairs)
    all_items = np.arange(np.max(dataset.data) + 1)

    if not os.path.exists("../data/All/test.negative_sample.csv"):
        np.random.seed(0)
        sampled_item_pools = {}
        for i in interactions:
            sampled_items = np.random.choice(all_items, size=999)

            filtered_sampled_items = []
            for j in sampled_items:
                if (i, j) in train_positive_pairs or (j, i) in train_positive_pairs or j in interactions[i]:
                    continue
                else:
                    filtered_sampled_items.append(j)

            filtered_sampled_items = set(filtered_sampled_items)
            while len(filtered_sampled_items) < 999:
                j = np.random.choice(all_items)
                while (i, j) in train_positive_pairs or (
                j, i) in train_positive_pairs or j in filtered_sampled_items or j in interactions[i]:
                    j = np.random.choice(all_items)
                filtered_sampled_items.add(j)

            filtered_sampled_items = list(filtered_sampled_items)
            sampled_item_pools[i] = filtered_sampled_items

        pd.DataFrame({
            'keys': list(sampled_item_pools.keys()),
            'values': list(sampled_item_pools.values())
        }).to_csv("../data/All/test.negative_sample.csv", index=False, sep=',')

    else:
        dataframe = pd.read_csv("../data/All/test.negative_sample.csv", sep=',')
        sampled_item_pools = {
            key: eval(value) for (key, value) in zip(dataframe['keys'].values, dataframe['values'].values)
        }

    def get_prediction_scores(test_pairs, filename):
        if not os.path.exists(filename):
            test_dataset = Dataset(
                path=cfg['path'],
                data=test_pairs,
                negative_sampling_rate=0,
                tokenizer=gpt2_tokenizer,
                debug=cfg['debug'],
                concat=True
            )

            test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=num_workers)

            all_predictions = []

            for batch in tqdm(test_loader):

                x, targets = batch

                prepared_batch = prepare_minibatch(x, targets, device, return_reverse_map=True, concat=True)

                x, mask, targets, reverse_map = prepared_batch

                # logits, z, _ = model(x)  # forward pass
                # idx1, idx2, x1, x2, targets = batch
                targets = targets.to(device)
                batch_size = targets.size(0)
                with torch.no_grad():
                    # batch = prepare_minibatch([x1, x2], targets, device, return_reverse_map=True, sort_by_length=False)
                    prepared_batch = prepare_minibatch(x, targets, device, return_reverse_map=True, concat=True,
                                                       return_idx=True)

                    idxes, x, mask, targets, reverse_map = prepared_batch

                    # if batch is None:
                    #     print("Idx1:", idx1)
                    #     print("Idx2:", idx2)
                    #     all_predictions.append(torch.ones(batch_size) * -9999)
                    #     continue

                    # idxes, x1, x2, mask1, mask2, targets = batch

                    try:
                        # logits = model([x1, x2], [mask1, mask2])[0]
                        logits, z, _ = model(x)
                        predictions = torch.ones(batch_size) * -9999
                        predictions[idxes] = logits[:, 1].cpu().data
                        all_predictions.append(predictions)

                    except:

                        # logits1 = model([x1[:batch_size//2], x2[:batch_size//2]], [mask1[:batch_size//2], mask2[:batch_size//2]])
                        logits1, z, _ = model(x[:batch_size // 2])
                        logits1 = logits1.cpu()
                        logits = model(x[batch_size // 2:])
                        logits = torch.cat([logits1, logits.cpu()])

                        predictions = torch.ones(batch_size) * -9999
                        predictions[idxes] = logits[:, 1].cpu()
                        all_predictions.append(predictions)

            all_predictions_scores = torch.cat(all_predictions).numpy()
            np.save(filename, all_predictions_scores)

        else:
            all_predictions_scores = np.load(filename)

        return all_predictions_scores

    positive_pairs = []
    for i in interactions:
        for j in interactions[i]:
            positive_pairs.append([i, j])
    positive_pairs = np.array(positive_pairs)
    positive_predictions = get_prediction_scores(positive_pairs,
                                                 filename=os.path.join(cfg['path'], "positive_scores.npy"))
    positive_prediction_mapping = {}

    negative_pairs = []
    for i in interactions:
        for j in sampled_item_pools[i]:
            negative_pairs.append([i, j])
    negative_pairs = np.array(negative_pairs)

    all_negative_prediction_scores = get_prediction_scores(negative_pairs, filename=os.path.join(cfg['save_path'],
                                                                                                 "negative_scores.npy"))

    count = 0
    for i in interactions:
        for j in interactions[i]:
            positive_prediction_mapping[(i, j)] = positive_predictions[count]
            count += 1

    topN = [5, 10, 20, 50]

    predicted_indices = []
    groundtruth = []
    for idx, i in enumerate(interactions.keys()):

        negative_prediction_scores = all_negative_prediction_scores[idx * 999: (idx + 1) * 999]

        for j in interactions[i]:

            positive_prediction_score = positive_prediction_mapping[(i, j)]
            indices = np.argsort(np.concatenate([np.array([positive_prediction_score]), negative_prediction_scores]))[
                      -max(topN):][::-1]
            groundtruth.append([0])
            if len(indices) == 1:
                pdb.set_trace()
            predicted_indices.append(indices.tolist())
        # predictions = all_predictions_scores[idx*1000:(idx+1)*1000]
        # indices = np.array(sampled_item_pools[i])[np.argsort(predictions)[-max(topN):][::-1]]
        # predicted_indices.append(indices.tolist())
        # groundtruth.append(interactions[i])
    precision, recall, NDCG, MRR = compute_acc(groundtruth, predicted_indices, topN)
    # print(precision, recall, NDCG, MRR)
    print("Recall:", recall)


def set_seed(seed):
    if seed == -1:
        seed = int(torch.randint(0, 2 ** 32 - 1, (1,)).item())

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def upper_category(noun):

    sets = [
        ['jean', 'pant', 'dress', 'sweatpant', 'panty', 'shorts', 'kilt', 'bloomer', 'culotte', 'skirt', 'trouser', 'leggings', 'overalls', 'jumpsuits', 'pant', 'jean', 'dungaree', 'gown', 'skate', 'satin', 'jean'],
        ['hat', 'cap', 'panama', 'deerstalker', 'beanie', 'visor', 'beret', 'pompom', 'fedora', 'wristlet', 'cloche', 'panama', 'bonnet'],
        ['pack', 'backpack', 'bag'],
        ['flip-flop', 'sneaker', 'pump', 'slipper', 'clog', 'espadrille', 'moccasin', 'ballerina', 'brogue', 'bootie', 'loafer', 'gumboot', 'slingback', 'shoes'],
        ['cardigan', 'blouse', 'jumper', 'lingerie', 'shirt', 'kimono', 'top', 'swimsuit', 't-shirt', 'hoodie', 'petit', 'robe', 'leotard', 'vest', 'blazer', 'Underwear', 'slip-on', 'coat', 'Knickerbocker', 'Gown']
    ]

    for idx, group in enumerate(sets):
        for word in group:
            if word in noun:
                return idx

    stemmer = PorterStemmer()
    return stemmer.stem(noun.split(" ")[-1].lower())

def saveprompts(cfg):
    cfg = vars(cfg)
    set_seed(cfg['seed'])

    for k, v in cfg.items():
        print("{:20} : {:10}".format(k, v))

    num_iterations = cfg["num_iterations"]
    print_every = cfg["print_every"]
    eval_every = cfg["eval_every"]
    batch_size = cfg["batch_size"]
    eval_batch_size = cfg.get("eval_batch_size", batch_size)
    topk = cfg['topk']

    if cfg['llm'] == 'bert':
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif cfg['llm'] == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    else:
        raise NotImplementedError

    print("Loading data")
    dataset = Dataset(
        path=cfg['path'],
        tokenizer=tokenizer,
        debug=cfg['debug'],
        concat=True,
        return_idx=True
    )

    set_seed(0)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # model = build_model(cfg)
    # initialize_model_(model)
    assert cfg['pretrained_weights'] is not None, "You should specify pretrained weights"
    save_path = cfg['save_path']
    model_ckpt = torch.load(cfg["pretrained_weights"], map_location=device)
    cfg = model_ckpt["cfg"]
    model = build_model(cfg)
    model.load_state_dict(model_ckpt["state_dict"])
    model.to(device)
    model.train()
    # model.eval()

    path = cfg['path']
    with open(f"{path}nouns.txt", 'r+') as file:
        nouns = file.read().strip().split("\n")
    file.close()

    prompts = []
    scores = []
    sentences = []

    with torch.no_grad():

        for idx, batch in tqdm(enumerate(data_loader)):

            if len(prompts) > 500: break

            item_idx1, item_idx2, x, targets = batch

            if targets[0] == 0:
                continue

            if nouns[item_idx1[0]] == "None" or len(nouns[item_idx1[0]]) == 0:
                continue
            if nouns[item_idx2[0]] == "None" or len(nouns[item_idx2[0]]) == 0:
                continue
            if upper_category(nouns[item_idx1[0]]) == upper_category(nouns[item_idx2[0]]):
                continue

            prepared_batch = prepare_minibatch(x, targets, device, return_reverse_map=True, concat=True)

            x, mask, targets, reverse_map = prepared_batch

            model.train()
            logits, z_train, _ = model(x)  # forward pass
            model.eval()
            logits, z_eval, _ = model(x)  # forward pass

            for i in range(len(logits)):

                split_idx = torch.where(x[i] == 102)[0].item()
                if 50257 in x[i]:
                    end_idx = torch.where(x[i] == 50257)[0][0].item()
                else:
                    end_idx = len(x[i])

                x1 = x[i][:split_idx]

                z1_train = z_train[i][:split_idx].cpu().numpy()
                z1_eval = z_eval[i][:split_idx].cpu().numpy()

                x2 = x[i][split_idx + 1:end_idx]

                z2_train = z_train[i][split_idx:].cpu().numpy()
                z2_eval = z_eval[i][split_idx:].cpu().numpy()

                sentence1 = tokenizer.decode(x1.cpu()).split(" ")
                sentence2 = tokenizer.decode(x2.cpu()).split(" ")

                blocks1 = get_blocks(sentence1, tokenizer)  # For instance, [1, 1, 3, 1]
                blocks2 = get_blocks(sentence2, tokenizer)

                blocks1 = np.cumsum(blocks1)  # For instance, [1, 2, 5, 6]
                blocks2 = np.cumsum(blocks2)

                blocks1 = [0] + blocks1.tolist()  # for instance, [0, 1, 2, 5, 6]
                blocks2 = [0] + blocks2.tolist()

                def get_attn(z1, z2):
                    new_attn1 = []
                    for idx1, b1 in enumerate(blocks1[1:]):
                        new_attn1.append(np.mean(z1[blocks1[idx1]: b1]))

                    new_attn2 = []
                    for idx2, b2 in enumerate(blocks2[1:]):
                        new_attn2.append(np.mean(z2[blocks2[idx2]: b2]))

                    new_attn1, new_attn2 = np.array(new_attn1), np.array(new_attn2)
                    return new_attn1, new_attn2

                new_attn1_train, new_attn2_train = get_attn(z1_train, z2_train)
                new_attn1_eval, new_attn2_eval = get_attn(z1_eval, z2_eval)

                if len(np.where(new_attn1_eval > 0)[0]) == 0 or len(np.where(new_attn2_eval > 0)[0]) == 0:
                    continue

                if np.isnan(new_attn1_train).any() or np.isnan(new_attn2_train).any():
                    continue

                words_1 = np.array(sentence1)[np.where(new_attn1_eval > 0)[0]].tolist()
                words_2 = np.array(sentence2)[np.where(new_attn2_eval > 0)[0]].tolist()

                prompts.append(" ".join(
                    [nouns[item_idx1[0]], "|", nouns[item_idx2[0]], "|", " ".join(words_1).lower(), "|",
                     " ".join(words_2).lower()]))

                new_attn1_train = new_attn1_train.tolist()
                new_attn2_train = new_attn2_train.tolist()
                scores.append("|".join([str(new_attn1_train), str(new_attn2_train)]))
                sentences.append(" ".join(sentence1) + "|" + " ".join(sentence2))

    pd.DataFrame(
        {
            'prompts': prompts,
            'scores': scores,
            'sentences': sentences
        }
    ).to_csv(f"{path}prompts_re_{cfg['selection']}.csv", index=False)

