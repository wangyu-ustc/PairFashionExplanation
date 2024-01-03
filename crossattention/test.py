import os
import pdb
import sys
import time
import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
import json
import argparse

from tqdm import tqdm
import torch
import torch.optim
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
# sys.path.append("/data2/zexue/debias_by_rationale")
from common.util import make_kv_string, build_model
from utils import \
    prepare_minibatch, print_parameters, \
    initialize_model_, get_device
from transformers import BertTokenizer
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader, Subset
from train import Dataset
from utils import train_valid_test_split, compute_acc
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns

device = get_device()
print("device:", device)


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


def get_scores(cfg):
    cfg = vars(cfg)
    set_seed(cfg['seed'])
    for k, v in cfg.items():
        print("{} : {}".format(k, v))
    num_iterations = cfg["num_iterations"]
    print_every = cfg["print_every"]
    eval_every = cfg["eval_every"]
    batch_size = cfg["batch_size"]
    eval_batch_size = cfg.get("eval_batch_size", batch_size)
    topk = cfg['topk']
    gpt2_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print("Loading data")
    dataset = Dataset(
        path=cfg['path'],
        tokenizer=gpt2_tokenizer,
        debug=cfg['debug'],
    )
    num_workers = multiprocessing.cpu_count() if cfg['num_workers'] == -1 else cfg['num_workers']
    data_loader = DataLoader(dataset, batch_size=1,
                             shuffle=False, num_workers=num_workers)
    # model = build_model(cfg)
    # initialize_model_(model)
    # assert cfg['pretrained_weights'] is not None, "You should specify pretrained weights"

    model = build_model(cfg)

    if cfg['pretrained_weights'] is not None:
        model_ckpt = torch.load(cfg["pretrained_weights"], map_location=device)
        cfg = model_ckpt["cfg"]
        model.load_state_dict(model_ckpt["state_dict"])
        print(f"load model weights from {model_ckpt}")
    save_path = cfg['save_path']
    model.to(device)
    model.eval()

    with torch.no_grad():
        # model.embed.weight.data.copy_(torch.from_numpy(vectors))
        if cfg["fix_emb"]:
            print("fixed word embeddings")
            model.embed.weight.requires_grad = False
        model.embed.weight[1] = 0.  # padding zero

    optimizer = Adam(model.parameters(), lr=cfg["lr"],
                     weight_decay=cfg["weight_decay"])
    iter_i = 0
    start = time.time()
    best_iter = 0

    model = model.to(device)

    # print model
    print(model)
    print_parameters(model)

    last_length = 0
    scores = []

    print("Length of data_loader:", len(data_loader))
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(data_loader)):

            x1, x2, targets = batch
            # x1, x2, mask1, mask2, targets, reverse_map1, reverse_map2 = prepare_minibatch([x1, x2], targets, device, return_reverse_map=True)
            prepared_batch = prepare_minibatch([x1, x2], targets, device, return_reverse_map=True)

            if prepared_batch is None:
                continue
            else:
                x1, x2, mask1, mask2, targets, reverse_map1, reverse_map2 = prepared_batch

            logits, attn = model((x1, x2), (mask1, mask2), (reverse_map1, reverse_map2), return_attn=True)

            attn = attn.squeeze(0).squeeze(-1)

            sentence1 = dataset.tz.decode(x1[0].cpu()).split(" ")
            sentence2 = dataset.tz.decode(x2[0].cpu()).split(" ")

            blocks1 = get_blocks(sentence1, dataset.tz)  # For instance, [1, 1, 3, 1]
            blocks2 = get_blocks(sentence2, dataset.tz)

            blocks1 = np.cumsum(blocks1)  # For instance, [1, 2, 5, 6]
            blocks2 = np.cumsum(blocks2)

            blocks1 = [0] + blocks1.tolist()  # for instance, [0, 1, 2, 5, 6]
            blocks2 = [0] + blocks2.tolist()

            new_attn = torch.zeros(len(sentence2), len(sentence1))

            for idx1, b1 in enumerate(blocks1[1:]):
                for idx2, b2 in enumerate(blocks2[1:]):
                    tmp_val = torch.mean(attn[blocks2[idx2]: b2, blocks1[idx1]: b1])
                    if np.isnan(tmp_val.cpu().numpy()):
                        pdb.set_trace()
                    new_attn[idx2, idx1] = tmp_val
            new_attn = new_attn.numpy()

            importance_scores = new_attn.reshape(-1)

            # # process new attn
            # importance_scores = new_attn.max(axis=0).tolist() + new_attn.max(axis=1).tolist()
            # importance_scores = torch.tensor(importance_scores)

            if topk > 0 and len(importance_scores) < topk:
                continue
            if topk > 0:
                scores.extend(torch.topk(importance_scores, topk)[0].tolist())
            else:
                scores.extend(importance_scores.tolist())

        json.dump(scores, open(save_path + f'./scores_{topk}_mean.json', "w"))


def elle_test(cfg):
    data = pd.read_csv("../data/statistics_item_feature.csv", sep=',')
    features = data['features'].values
    true_features = [eval(feature) for feature in features if len(eval(feature)) == 2]

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    cfg = vars(cfg)
    # assert cfg['pretrained_weights'] is not None, "You should specify pretrained weights"
    model = build_model(cfg)

    if cfg['pretrained_weights'] is not None:
        model_ckpt = torch.load(cfg["pretrained_weights"], map_location=device)
        print(f"load model weights from ", cfg["pretrained_weights"])
        cfg = model_ckpt["cfg"]
        model.load_state_dict(model_ckpt["state_dict"])
    save_path = cfg['save_path']
    model.to(device)
    model.eval()

    scores = []

    with torch.no_grad():
        for feat1, feat2 in tqdm(true_features):

            x1 = tokenizer(feat1)['input_ids'][1:-1]
            x2 = tokenizer(feat2)['input_ids'][1:-1]

            # shift x1 and x2
            # tmp = x1
            # x1 = x2
            # x2 = tmp

            # max_length = max(len(x1), len(x2))
            if len(x1) == 0 or len(x2) == 0:
                print("X1:", x1)
                print("x2:", x2)
                print("feat1:", feat1)
                print("feat2:", feat2)
                continue
            # x1 += [50257] * (max_length - len(x1))
            # x2 += [50257] * (max_length - len(x2))
            x1, x2 = torch.tensor(x1).unsqueeze(0).to(device), torch.tensor(x2).unsqueeze(0).to(device)

            mask1 = x1 != 50257
            mask2 = x2 != 50257
            try:
                logits, attn = model((x1, x2), (mask1, mask2), return_attn=True)
            except:
                pdb.set_trace()

            attn = attn.squeeze(0).squeeze(-1)

            sentence1 = tokenizer.decode(x1[0].cpu()).split(" ")
            sentence2 = tokenizer.decode(x2[0].cpu()).split(" ")

            blocks1 = get_blocks(sentence1, tokenizer)  # For instance, [1, 1, 3, 1]
            blocks2 = get_blocks(sentence2, tokenizer)

            blocks1 = np.cumsum(blocks1)  # For instance, [1, 2, 5, 6]
            blocks2 = np.cumsum(blocks2)

            blocks1 = [0] + blocks1.tolist()  # for instance, [0, 1, 2, 5, 6]
            blocks2 = [0] + blocks2.tolist()

            new_attn = torch.zeros(len(sentence2), len(sentence1))
            for idx1, b1 in enumerate(blocks1[1:]):
                for idx2, b2 in enumerate(blocks2[1:]):
                    try:
                        tmp_val = torch.mean(attn[blocks2[idx2]: b2, blocks1[idx1]: b1])
                    except:
                        pdb.set_trace()
                    new_attn[idx2, idx1] = tmp_val

            new_attn = new_attn.numpy()

            # mean
            importance_scores = new_attn.reshape(-1)

            ##### just get max or mean scores in attn
            # scores.append(new_attn.max().item())

            ##### Use important scores
            # importance_scores = new_attn.reshape(-1)

            # process new attn
            # importance_scores = new_attn.max(axis=0).tolist() + new_attn.max(axis=1).tolist()
            # importance_scores = torch.tensor(importance_scores)

            scores.extend(importance_scores.tolist())

        json.dump(scores, open(save_path + f'elle_scores_mean.json', "w"))


def test_recall(cfg):
    # cfg = get_args()
    cfg = vars(cfg)

    set_seed(cfg['seed'])

    for k, v in cfg.items():
        print("{} : {}".format(k, v))

    num_iterations = cfg["num_iterations"]
    print_every = cfg["print_every"]
    eval_every = cfg["eval_every"]
    batch_size = cfg["batch_size"]
    eval_batch_size = cfg.get("eval_batch_size", batch_size)

    # gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    print("Loading data")
    dataset = Dataset(
        path=cfg['path'],
        tokenizer=gpt2_tokenizer,
        debug=cfg['debug']
    )

    train_idxes, valid_idxes, test_idxes = train_valid_test_split(np.arange(len(dataset)), 0.8, 0.1, 0.1)
    train_idxes = np.concatenate([train_idxes, valid_idxes])

    train_idxes = train_idxes[np.where(train_idxes < len(dataset.data))]
    test_idxes = test_idxes[np.where(test_idxes < len(dataset.data))]

    train_positive_pairs = {
        (i, j) for (i, j) in dataset.data[train_idxes]
    }

    num_workers = multiprocessing.cpu_count() if cfg['num_workers'] == -1 else cfg['num_workers']

    model = build_model(cfg)
    initialize_model_(model)

    assert cfg['pretrained_weights'] is not None, "You should specify pretrained weights"

    model_ckpt = torch.load(cfg["pretrained_weights"], map_location=device)
    cfg = model_ckpt["cfg"]
    model = build_model(cfg)
    model.load_state_dict(model_ckpt["state_dict"], strict=False)
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
                return_idx=True,
                debug=cfg['debug']
            )

            test_loader = DataLoader(test_dataset, batch_size=256 * 2, shuffle=False, num_workers=num_workers)

            all_predictions = []

            for batch in tqdm(test_loader):

                idx1, idx2, x1, x2, targets = batch
                targets = targets.to(device)
                batch_size = targets.size(0)
                with torch.no_grad():
                    batch = prepare_minibatch([x1, x2], targets, device, return_reverse_map=True, sort_by_length=False)

                    if batch is None:
                        print("Idx1:", idx1)
                        print("Idx2:", idx2)
                        all_predictions.append(torch.ones(batch_size) * -9999)
                        continue

                    idxes, x1, x2, mask1, mask2, targets = batch

                    try:
                        logits = model([x1, x2], [mask1, mask2])[0]
                        predictions = torch.ones(batch_size) * -9999
                        predictions[idxes] = logits[:, 1].cpu().data
                        all_predictions.append(predictions)

                    except:

                        logits1 = model([x1[:batch_size // 2], x2[:batch_size // 2]],
                                        [mask1[:batch_size // 2], mask2[:batch_size // 2]])
                        logits1 = logits1.cpu()
                        logits = model([x1[batch_size // 2:], x2[batch_size // 2:]],
                                       [mask1[batch_size // 2:], mask2[batch_size // 2:]])
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
    positive_predictions = get_prediction_scores(positive_pairs, filename='./logs/All_fix_emb/positive_scores.npy')
    positive_prediction_mapping = {}

    negative_pairs = []
    for i in interactions:
        for j in sampled_item_pools[i]:
            negative_pairs.append([i, j])
    negative_pairs = np.array(negative_pairs)

    all_negative_prediction_scores = get_prediction_scores(negative_pairs,
                                                           filename='./logs/All_fix_emb/negative_scores.npy')

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


# def test_recall(cfg):
#      # cfg = get_args()
#     cfg = vars(cfg)

#     set_seed(cfg['seed'])

#     for k, v in cfg.items():
#         print("{} : {}".format(k, v))

#     num_iterations = cfg["num_iterations"]
#     print_every = cfg["print_every"]
#     eval_every = cfg["eval_every"]
#     batch_size = cfg["batch_size"]
#     eval_batch_size = cfg.get("eval_batch_size", batch_size)

#     # gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#     gpt2_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#     gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

#     print("Loading data")
#     dataset = Dataset(
#         path = cfg['path'],
#         tokenizer=gpt2_tokenizer,
#         debug=cfg['debug']
#     )

#     train_idxes, valid_idxes, test_idxes = train_valid_test_split(np.arange(len(dataset)), 0.8, 0.1, 0.1)
#     train_idxes = np.concatenate([train_idxes, valid_idxes])

#     train_idxes = train_idxes[np.where(train_idxes < len(dataset.data))]
#     test_idxes = test_idxes[np.where(test_idxes < len(dataset.data))]

#     train_positive_pairs = {
#         (i, j) for (i, j) in dataset.data[train_idxes]
#     }

#     num_workers = multiprocessing.cpu_count() if cfg['num_workers'] == -1 else cfg['num_workers']

#     model = build_model(cfg)
#     initialize_model_(model)

#     assert cfg['pretrained_weights'] is not None, "You should specify pretrained weights"

#     model_ckpt = torch.load(cfg["pretrained_weights"], map_location=device)
#     cfg = model_ckpt["cfg"]
#     model = build_model(cfg)
#     model.load_state_dict(model_ckpt["state_dict"])
#     model.to(device)

#     test_positive_pairs = dataset.data[test_idxes]

#     def construct_interaction_dict(test_positive_pairs):
#         interactions = {}

#         def add_item_to_list(i, j):
#             if i in interactions:
#                 interactions[i].append(j)
#             else:
#                 interactions[i] = [j]


#         for (i, j) in test_positive_pairs:
#             add_item_to_list(i, j)
#             add_item_to_list(j, i)

#         for key in interactions:
#             interactions[key] = list(set(interactions[key]))

#         return interactions

#     train_interactions = construct_interaction_dict(dataset.data[train_idxes])
#     interactions = construct_interaction_dict(test_positive_pairs)

#     all_items = np.arange(np.max(dataset.data) + 1)


#     # if not os.path.exists("../data/All/test.negative.csv"):
#     #     np.random.seed(0)
#     #     sampled_item_pools = {}
#     #     for i in interactions:
#     #         sampled_items = all_items.copy()
#     #         if i in train_interactions:
#     #             exclude_idxes = train_interactions[i]
#     #             left_idxes = set(list(np.arange(len(sampled_items)))) - set(list(exclude_idxes))
#     #             sampled_items = sampled_items[np.array(list(left_idxes))]
#     #         sampled_item_pools[i] = sampled_items

#     #         # sampled_items = np.random.choice(all_items, size=1000)

#     #         # filtered_sampled_items = []
#     #         # for j in sampled_items:
#     #         #     if (i, j) in train_positive_pairs or (j, i) in train_positive_pairs:
#     #         #         continue
#     #         #     else:
#     #         #         filtered_sampled_items.append(j)

#     #         # filtered_sampled_items = set(filtered_sampled_items)
#     #         # while len(filtered_sampled_items) < 1000:
#     #         #     j = np.random.choice(all_items)
#     #         #     while (i, j) in train_positive_pairs or (j, i) in train_positive_pairs or j in filtered_sampled_items:
#     #         #         j = np.random.choice(all_items)
#     #         #     filtered_sampled_items.add(j)

#     #         # filtered_sampled_items = list(filtered_sampled_items)
#     #         # sampled_item_pools[i] = filtered_sampled_items

#     #     # json.dump(sampled_item_pools, open("../data/All/test.negative.json", 'w'))
#     #     pd.DataFrame({
#     #         'keys': list(sampled_item_pools.keys()),
#     #         'values': list(sampled_item_pools.values())
#     #     }).to_csv("../data/All/test.negative.csv", index=False, sep=',')

#     # else:
#     #     dataframe = pd.read_csv("../data/All/test.negative.csv", sep=',')
#     #     sampled_item_pools = {
#     #         key: eval(value) for (key, value) in zip(dataframe['keys'].values, dataframe['values'].values)
#     #     }

#     # all_test_pairs = []
#     # for i in interactions:
#     #     for j in sampled_item_pools[i]:
#     #         all_test_pairs.append([i, j])
#     # all_test_pairs = np.array(all_test_pairs)

#     class pairs():
#         def __init__(self, interactions, all_items):
#             self.items = list(interactions.keys())
#             self.all_items = all_items
#             self.length = len(self.items) * len(self.all_items)
#             self.l2 =  len(self.all_items)

#         def __call__(self, i):
#             return (self.items[i//self.l2], i%self.l2)

#         def __len__(self):
#             return self.length


#     all_test_pairs = []
#     print("Start building pairs")
#     all_test_pairs = pairs(interactions, all_items)
#     # for i in tqdm(interactions):
#     #     all_test_pairs.extend(np.stack([np.ones(len(all_items)) * i, all_items]).transpose().astype(int).tolist())

#     pdb.set_trace()

#     test_dataset = Dataset(
#         path=cfg['path'],
#         data=all_test_pairs,
#         negative_sampling_rate=0,
#         tokenizer=gpt2_tokenizer,
#         return_idx=True,
#         debug=cfg['debug']
#     )

#     test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=num_workers)

#     all_predictions = []

#     for batch in tqdm(test_loader):

#         idx1, idx2, x1, x2, targets = batch
#         targets = targets.to(device)
#         batch_size = targets.size(0)
#         with torch.no_grad():
#             batch = prepare_minibatch([x1, x2], targets, device, return_reverse_map=True)
#             x1, x2, mask1, mask2, targets, reverse_map1, reverse_map2 = batch
#             logits, attn = model([x1, x2], [mask1, mask2], [reverse_map1, reverse_map2], return_attn=True)

#         # for idx1i, idx2i, predi in zip(idx1, idx2, logits[:,1]):
#         #     all_predictions[(idx1i.item(), idx2i.item())] = predi.item()
#         all_predictions.append(logits[:, 1].data)

#     all_predictions_scores = torch.cat(all_predictions).numpy()
#     # all_predictions_scores = np.array(list(all_predictions.values()))

#     def test_all_pairs(topk=5):

#         count = 0

#         predicted_indices = []
#         groundtruth = []
#         for idx, i in enumerate(interactions.keys()):

#             predictions = all_predictions_scores[idx*len(all_items):(idx+1)*len(all_items)]
#             if i in train_interactions:
#                 predictions[train_interactions[i]] = -9999

#             # if idx < 5:
#             #     # do some test
#             #     predictions_for_i = []
#             #     for j in sampled_item_pools[i]:
#             #         predictions_for_i.append(all_predictions[(i, j)])

#             #     assert (np.array(predictions_for_i) == predictions).all()

#             indices = np.argsort(predictions)[-topk:]
#             predicted_indices.append(indices.tolist())
#             groundtruth.append(interactions[i])

#         precision, recall, NDCG, MRR = compute_acc(GroundTruth, predictedIndices, top_k)
#         return precision, recall, NDCG, MRR

#     for topk in [5, 10, 20, 50]:
#         print(f"Top {topk}")
#         print(test_all_pairs(topk))


def savewords(cfg):
    """
    Main training loop.
    """

    # cfg = get_args()
    cfg = vars(cfg)

    set_seed(cfg['seed'])

    for k, v in cfg.items():
        print("{} : {}".format(k, v))

    num_iterations = cfg["num_iterations"]
    print_every = cfg["print_every"]
    eval_every = cfg["eval_every"]
    batch_size = cfg["batch_size"]
    eval_batch_size = cfg.get("eval_batch_size", batch_size)

    # gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    print("Loading data")
    dataset = Dataset(
        path=cfg['path'],
        tokenizer=gpt2_tokenizer,
        debug=cfg['debug'],
    )

    num_workers = multiprocessing.cpu_count() if cfg['num_workers'] == -1 else cfg['num_workers']

    data_loader = DataLoader(dataset, batch_size=1,
                             shuffle=False, num_workers=num_workers)

    model = build_model(cfg)
    initialize_model_(model)

    assert cfg['pretrained_weights'] is not None, "You should specify pretrained weights"

    save_path = cfg['save_path']

    model_ckpt = torch.load(cfg["pretrained_weights"], map_location=device)
    cfg = model_ckpt["cfg"]
    model = build_model(cfg)
    model.load_state_dict(model_ckpt["state_dict"])
    model.to(device)
    model.eval()

    # Build model

    with torch.no_grad():
        # model.embed.weight.data.copy_(torch.from_numpy(vectors))
        if cfg["fix_emb"]:
            print("fixed word embeddings")
            model.embed.weight.requires_grad = False
        model.embed.weight[1] = 0.  # padding zero

    optimizer = Adam(model.parameters(), lr=cfg["lr"],
                     weight_decay=cfg["weight_decay"])

    iter_i = 0
    start = time.time()
    best_iter = 0

    model = model.to(device)

    # print model
    print(model)
    print_parameters(model)

    words = defaultdict(lambda: 0)

    last_length = 0

    print("Length of data_loader:", len(data_loader))

    for idx, batch in enumerate(data_loader):

        x1, x2, targets = batch
        with torch.no_grad():

            prepared_batch = prepare_minibatch([x1, x2], targets, device, return_reverse_map=True)

            if prepared_batch is None:
                continue
            else:
                x1, x2, mask1, mask2, targets, reverse_map1, reverse_map2 = prepared_batch

            if len(x1) == 0:
                print("Zero length")
                sys.stdout.flush()
                continue

            logits, attn = model((x1, x2), (mask1, mask2), (reverse_map1, reverse_map2), return_attn=True)

            # x1: (1, length), x2: (1, length)
            # attn.shape: (x2.shape[1], x1.shape[1])
            attn = attn.data.cpu().squeeze(0).squeeze(-1)

            sentence1 = dataset.tz.decode(x1[0].cpu()).split(" ")
            sentence2 = dataset.tz.decode(x2[0].cpu()).split(" ")

            blocks1 = get_blocks(sentence1, dataset.tz)  # For instance, [1, 1, 3, 1]
            blocks2 = get_blocks(sentence2, dataset.tz)

            blocks1 = np.cumsum(blocks1)  # For instance, [1, 2, 5, 6]
            blocks2 = np.cumsum(blocks2)

            blocks1 = [0] + blocks1.tolist()  # for instance, [0, 1, 2, 5, 6]
            blocks2 = [0] + blocks2.tolist()

            new_attn = torch.zeros(len(sentence2), len(sentence1))

            for idx1, b1 in enumerate(blocks1[1:]):
                for idx2, b2 in enumerate(blocks2[1:]):
                    try:
                        tmp_val = torch.mean(attn[blocks2[idx2]: b2, blocks1[idx1]: b1])
                    except:
                        pdb.set_trace()
                    new_attn[idx2, idx1] = tmp_val

            idx = torch.argmax(new_attn.sum(dim=0))
            words[sentence1[idx].lower()] += 1

            idx = torch.argmax(new_attn.sum(dim=1))
            words[sentence2[idx].lower()] += 1

        length = len(words)

        if length > last_length:
            print(f"Words {last_length} -> {length}")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open(save_path + '/words.json', 'w') as file:
                # file.write("\n".join(words))
                json.dump(words, file)
            file.close()

        last_length = length


def visualize(cfg):
    """
    Main training loop.
    """

    # cfg = get_args()
    cfg = vars(cfg)

    set_seed(cfg['seed'])

    for k, v in cfg.items():
        print("{} : {}".format(k, v))

    num_iterations = cfg["num_iterations"]
    print_every = cfg["print_every"]
    eval_every = cfg["eval_every"]
    batch_size = cfg["batch_size"]
    eval_batch_size = cfg.get("eval_batch_size", batch_size)

    # gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    print("Loading data")
    dataset = Dataset(
        path=cfg['path'],
        tokenizer=gpt2_tokenizer,
        debug=cfg['debug'],
    )

    train_idxes, valid_idxes, test_idxes = train_valid_test_split(
        np.arange(len(dataset)) * (1 + dataset.negative_sampling_rate), 0.8, 0.1, 0.1)
    num_workers = multiprocessing.cpu_count() if cfg['num_workers'] == -1 else cfg['num_workers']

    test_loader = DataLoader(Subset(dataset, indices=test_idxes), batch_size=1,
                             shuffle=False, num_workers=num_workers)

    model = build_model(cfg)
    initialize_model_(model)

    assert cfg['pretrained_weights'] is not None, "You should specify pretrained weights"

    model_ckpt = torch.load(cfg["pretrained_weights"], map_location=device)
    cfg = model_ckpt["cfg"]
    model = build_model(cfg)
    model.load_state_dict(model_ckpt["state_dict"])
    model.to(device)
    model.eval()

    iter_i = 0
    start = time.time()
    best_iter = 0

    model = model.to(device)

    # print model
    print(model)
    print_parameters(model)

    for idx, batch in enumerate(test_loader):

        x1, x2, targets = batch
        with torch.no_grad():
            x1, x2, mask1, mask2, targets, reverse_map1, reverse_map2 = prepare_minibatch([x1, x2], targets, device,
                                                                                          return_reverse_map=True)
            z1, z2, logits, attn = model([x1, x2], [mask1, mask2], [reverse_map1, reverse_map2], return_attn=True)

            # x1: (1, length), x2: (1, length)
            # attn.shape: (x2.shape[1], x1.shape[1])
            attn = attn.data.cpu().squeeze(0).squeeze(-1)
            attn = attn / torch.sum(attn)

            sentence1 = dataset.tz.decode(x1[0].cpu()).split(" ")
            sentence2 = dataset.tz.decode(x2[0].cpu()).split(" ")

            blocks1 = get_blocks(sentence1, dataset.tz)  # For instance, [1, 1, 3, 1]
            blocks2 = get_blocks(sentence2, dataset.tz)

            blocks1 = np.cumsum(blocks1)  # For instance, [1, 2, 5, 6]
            blocks2 = np.cumsum(blocks2)

            blocks1 = [0] + blocks1.tolist()  # for instance, [0, 1, 2, 5, 6]
            blocks2 = [0] + blocks2.tolist()

            new_attn = torch.zeros(len(sentence2), len(sentence1))
            try:
                for idx1, b1 in enumerate(blocks1[1:]):
                    for idx2, b2 in enumerate(blocks2[1:]):
                        new_attn[idx2, idx1] = torch.mean(attn[blocks2[idx2]: b2, blocks1[idx1]: b1])
            except:
                pdb.set_trace()

            fig, ax = plt.subplots(1, 1)
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            sns.heatmap(new_attn, cmap=cmap, vmax=.3, center=0,
                        square=True, linewidths=.5, cbar_kws={"shrink": .5})
            ax.set_xticks(np.arange(len(sentence1)) + 0.5)
            ax.set_yticks(np.arange(len(sentence2)) + 0.5)
            ax.set_xticklabels(sentence1, rotation=80, fontsize=10)
            ax.set_yticklabels(sentence2, fontsize=10, rotation=0)
            plt.tight_layout()

            plt.savefig(f"./figures/fig_{targets.item()}_pred_{logits.squeeze()[0] < logits.squeeze()[1]}_{idx}.png")
            plt.close()


def saveprompts(cfg):
    """
    Main training loop.
    """

    # cfg = get_args()
    cfg = vars(cfg)

    set_seed(cfg['seed'])

    for k, v in cfg.items():
        print("{} : {}".format(k, v))

    assert cfg['pretrained_weights'] is not None, "You should specify pretrained weights"

    save_path = cfg['save_path']

    model = build_model(cfg)
    model_ckpt = torch.load(cfg["pretrained_weights"], map_location=device)
    cfg = model_ckpt["cfg"]
    model.load_state_dict(model_ckpt["state_dict"])
    model.to(device)
    model.eval()

    num_iterations = cfg["num_iterations"]
    print_every = cfg["print_every"]
    eval_every = cfg["eval_every"]
    batch_size = cfg["batch_size"]
    eval_batch_size = cfg.get("eval_batch_size", batch_size)

    # gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    print("Loading data")
    dataset = Dataset(
        path=cfg['path'],
        tokenizer=gpt2_tokenizer,
        debug=cfg['debug'],
        return_idx=True
    )

    num_workers = multiprocessing.cpu_count() if cfg['num_workers'] == -1 else cfg['num_workers']

    data_loader = DataLoader(dataset, batch_size=1,
                             shuffle=False, num_workers=num_workers)

    model = build_model(cfg)
    initialize_model_(model)

    path = cfg['path']
    with open(f"{path}nouns.txt", 'r+') as file:
        nouns = file.read().strip().split("\n")
    file.close()

    # Build model

    with torch.no_grad():
        # model.embed.weight.data.copy_(torch.from_numpy(vectors))
        if cfg["fix_emb"]:
            print("fixed word embeddings")
            model.embed.weight.requires_grad = False
        model.embed.weight[1] = 0.  # padding zero

    optimizer = Adam(model.parameters(), lr=cfg["lr"],
                     weight_decay=cfg["weight_decay"])

    iter_i = 0
    start = time.time()
    best_iter = 0

    model = model.to(device)

    # print model
    print(model)
    print_parameters(model)

    # words = set()
    prompts = []
    labels = []

    last_length = 0

    print("Length of data_loader:", len(data_loader))

    with open(f"{path}prompts.txt", 'w+') as prompts_file:
        with open(f"{path}prompts_targets.txt", 'w+') as target_file:
            for idx, batch in enumerate(data_loader):

                item_idx1, item_idx2, x1, x2, targets = batch
                if targets[0] == 0: continue
                with torch.no_grad():

                    prepared_batch = prepare_minibatch([x1, x2], targets, device, return_reverse_map=True)

                    if prepared_batch is None:
                        continue
                    else:
                        x1, x2, mask1, mask2, targets, reverse_map1, reverse_map2 = prepared_batch

                    if len(x1) == 0:
                        print("Zero length")
                        sys.stdout.flush()
                        continue

                    logits, attn = model((x1, x2), (mask1, mask2), (reverse_map1, reverse_map2), return_attn=True)

                    # x1: (1, length), x2: (1, length)
                    # attn.shape: (x2.shape[1], x1.shape[1])
                    attn = attn.data.cpu().squeeze(0).squeeze(-1)

                    sentence1 = dataset.tz.decode(x1[0].cpu()).split(" ")
                    sentence2 = dataset.tz.decode(x2[0].cpu()).split(" ")

                    blocks1 = get_blocks(sentence1, dataset.tz)  # For instance, [1, 1, 3, 1]
                    blocks2 = get_blocks(sentence2, dataset.tz)

                    blocks1 = np.cumsum(blocks1)  # For instance, [1, 2, 5, 6]
                    blocks2 = np.cumsum(blocks2)

                    blocks1 = [0] + blocks1.tolist()  # for instance, [0, 1, 2, 5, 6]
                    blocks2 = [0] + blocks2.tolist()

                    new_attn = torch.zeros(len(sentence2), len(sentence1))

                    for idx1, b1 in enumerate(blocks1[1:]):
                        for idx2, b2 in enumerate(blocks2[1:]):
                            try:
                                tmp_val = torch.mean(attn[blocks2[idx2]: b2, blocks1[idx1]: b1])
                            except:
                                pdb.set_trace()
                            new_attn[idx2, idx1] = tmp_val

                    feature_idx1 = torch.argsort(new_attn.sum(dim=0))
                    feature_idx2 = torch.argsort(new_attn.sum(dim=1))
                    try:
                        if nouns[item_idx1[0]] == "None" or len(nouns[item_idx1[0]]) == 0:
                            continue
                        if nouns[item_idx2[0]] == "None" or len(nouns[item_idx2[0]]) == 0:
                            continue
                    except:
                        pdb.set_trace()

                    words_1 = np.array(sentence1)[feature_idx1[-1:]].tolist()
                    if not isinstance(words_1, list):
                        words_1 = [words_1]

                    words_2 = np.array(sentence2)[feature_idx2[-1:]].tolist()
                    if not isinstance(words_2, list):
                        words_2 = [words_2]

                    try:
                        prompts_file.write(" ".join(
                            [nouns[item_idx1[0]], "|", nouns[item_idx2[0]], "|", " ".join(words_1).lower(), "|",
                             " ".join(words_2).lower()]) + '\n')
                    except:
                        print(nouns[item_idx1[0]])
                        print(nouns[item_idx2[0]])
                        print(sentence1)
                        print(sentence2)

                    target_file.write(str(targets[0].item()) + '\n')


def test(cfg):
    """
    Main training loop.
    """

    # cfg = get_args()
    cfg = vars(cfg)

    set_seed(cfg['seed'])

    for k, v in cfg.items():
        print("{} : {}".format(k, v))

    num_iterations = cfg["num_iterations"]
    print_every = cfg["print_every"]
    eval_every = cfg["eval_every"]
    batch_size = cfg["batch_size"]
    eval_batch_size = cfg.get("eval_batch_size", batch_size)

    # gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    print("Loading data")
    dataset = Dataset(
        path=cfg['path'],
        tokenizer=gpt2_tokenizer,
        debug=cfg['debug']
    )

    train_idxes, valid_idxes, test_idxes = train_valid_test_split(
        np.arange(len(dataset)) * (1 + dataset.negative_sampling_rate), 0.8, 0.1, 0.1)
    num_workers = multiprocessing.cpu_count() if cfg['num_workers'] == -1 else cfg['num_workers']

    val_loader = DataLoader(Subset(dataset, indices=valid_idxes), batch_size=cfg['batch_size'], shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(Subset(dataset, indices=test_idxes), batch_size=cfg['batch_size'],
                             shuffle=False, num_workers=num_workers)

    # writer = SummaryWriter(log_dir=cfg["save_path"])  # TensorBoard

    model = build_model(cfg)
    initialize_model_(model)

    assert cfg['pretrained_weights'] is not None, "You should specify pretrained weights"

    model_ckpt = torch.load(cfg["pretrained_weights"], map_location=device)
    cfg = model_ckpt["cfg"]
    model = build_model(cfg)
    model.load_state_dict(model_ckpt["state_dict"])
    model.to(device)

    pdb.set_trace()

    # Build model

    with torch.no_grad():
        # model.embed.weight.data.copy_(torch.from_numpy(vectors))
        if cfg["fix_emb"]:
            print("fixed word embeddings")
            model.embed.weight.requires_grad = False
        model.embed.weight[1] = 0.  # padding zero

    optimizer = Adam(model.parameters(), lr=cfg["lr"],
                     weight_decay=cfg["weight_decay"])

    start = time.time()
    best_iter = model_ckpt['best_iter']

    model = model.to(device)

    # print model
    print(model)
    print_parameters(model)

    print("# Evaluating")
    dev_eval = evaluate_single(
        model, data_loader=val_loader, batch_size=eval_batch_size, tokenizer=gpt2_tokenizer,
        device=device)
    test_eval = evaluate_single(
        model, data_loader=test_loader, batch_size=eval_batch_size, tokenizer=gpt2_tokenizer,
        device=device)

    print("best model iter {:d}: "
            "dev {} test {}".format(
        best_iter,
        make_kv_string(dev_eval),
        make_kv_string(test_eval)))

    # save result
    result_path = os.path.join(cfg["save_path"], f"{cfg['label']}_results.json")

    cfg["best_iter"] = best_iter

    for k, v in dev_eval.items():
        cfg["dev_" + k] = v
        # writer.add_scalar('best/dev/' + k, v, iter_i)

    for k, v in test_eval.items():
        print("test", k, v)
        cfg["test_" + k] = v
        # writer.add_scalar('best/test/' + k, v, iter_i)

    # writer.close()

    with open(result_path, mode="w") as f:
        json.dump(cfg, f)