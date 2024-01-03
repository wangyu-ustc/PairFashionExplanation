import os
import pdb
import sys
import time
import numpy as np
from collections import OrderedDict
import json
import argparse

from nltk.corpus import words
import torch
import torch.optim
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from common.util import make_kv_string
from common.util import build_model
from utils import prepare_minibatch, print_parameters, \
    initialize_model_, get_device
from transformers import BertTokenizer
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader, Subset
# from folds import Subset
from utils import train_valid_test_split, evaluate
from dataset import Dataset
import multiprocessing

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


def filter_desc(desc):
    new_desc = []
    all_words = set(words.words())
    for word in desc.split(" "):
        if word in all_words:
            new_desc.append(word)
    desc = " ".join(new_desc)
    return desc

class node():
    
    def __init__(self, value):
        
        self.value = value
        self.children = []

    def search_over_values(self, value):
        if self.value == value:
            return self
        for child in self.children:
            target = child.search_over_values(value)
            if target:
                return target
        return None

    def return_leaf_nodes(self):
        if self.children is None:
            return self
        else:
            leaf_nodes = []
            for child in self.chidren:
                leaf_nodes += child.return_leaf_nodes()
            return leaf_nodes
    
    def return_specific_layer_nodes(self, current_depth, target_depth):
        if current_depth == target_depth:
            return [self]
        else:
            nodes = []
            for child in self.children:
                nodes.extend(child.return_specific_layer_nodes(current_depth+1, target_depth))
            return nodes

def create_tree_from_dict(dictionary):

    if isinstance(dictionary, dict):

        root = node(list(dictionary.keys())[0])
        for child_dict in list(dictionary.values())[0]:
            root.children.append(create_tree_from_dict(child_dict))

    else:
        root = node(dictionary)
    
    return root


def build_tree_from_json(file_name):
    with open(file_name, "r+") as file:
        dictionary = json.load(file)
    return create_tree_from_dict(dictionary)



    
def check_grad(model):
    for param in model.parameters():
        if param.grad is not None:
            print(torch.isnan(param.grad).any())
    


def train(cfg):
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

    if cfg['llm'] == 'bert':
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif cfg['llm'] == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    else:
        raise NotImplementedError

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print("Loading data")
    dataset = Dataset(
        path = cfg['path'],
        tokenizer=tokenizer,
        debug=cfg['debug'],
    )

    train_idxes, valid_idxes, test_idxes = train_valid_test_split(np.arange(len(dataset)), 0.8, 0.1, 0.1)
    num_workers = multiprocessing.cpu_count() if cfg['num_workers'] == -1 else cfg['num_workers']

    train_loader = DataLoader(Subset(dataset, indices=train_idxes), batch_size=cfg['batch_size'], shuffle=True, num_workers=num_workers)

    val_loader = DataLoader(Subset(dataset, indices=valid_idxes), batch_size=cfg['batch_size'], shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(Subset(dataset, indices=test_idxes), batch_size=cfg['batch_size'],
                             shuffle=False, num_workers=num_workers)

    writer = SummaryWriter(log_dir=cfg["save_path"])  # TensorBoard

    model = build_model(cfg)
    initialize_model_(model)

    with torch.no_grad():
        # model.embed.weight.data.copy_(torch.from_numpy(vectors))
        if cfg["fix_emb"]:
            print("fixed word embeddings")
            model.embed.weight.requires_grad = False
        model.embed.weight[1] = 0.  # padding zero
    
    iter_i = 0
    train_loss = 0.
    print_num = 0
    start = time.time()
    losses = []
    accuracies = []
    best_eval = 1.0e9
    best_iter = 0

    model = model.to(device)

    if cfg['pretrained_weights'] is not None:
        model.load_state_dict(torch.load(cfg['pretrained_weights'])['state_dict'], strict=False)
        print("Loaded model from {}".format(cfg['pretrained_weights']))
    
    optimizer = Adam(model.parameters(), lr=cfg["lr"],
                     weight_decay=cfg["weight_decay"])

    # print model
    print(model)
    print_parameters(model)

    iters_per_epoch = len(train_loader)

    epoch = 0
    while True:  # when we run out of examples, shuffle and continue

        for idx, batch in enumerate(train_loader):

            model.train()

            x1, x2, targets = batch
            prepared_batch = prepare_minibatch([x1, x2], targets, device, return_reverse_map=True)

            if prepared_batch is None: continue
            else:
                x1, x2, mask1, mask2, targets, reverse_map1, reverse_map2 = prepared_batch

            if len(x1) == 0:
                print("Zero length")
                sys.stdout.flush()
                continue

            logits, lasso_penalty = model((x1, x2), (mask1, mask2), (reverse_map1, reverse_map2), 
                    return_attn=False, return_lasso=True)
            loss = model.get_loss(logits, targets)

            model.zero_grad()  # erase previous gradients

            loss += lasso_penalty

            train_loss += loss.item()
            loss.backward()

            optimizer.step()

            print_num += 1
            iter_i += 1

            # print info
            if iter_i % print_every == 0:

                train_loss = train_loss / print_every
                writer.add_scalar('train/loss', train_loss, iter_i)

                min_elapsed = (time.time() - start) // 60
                print("Epoch %r Iter %r/%r time=%dm loss=%.4f %.4f" %
                      (epoch, idx, iters_per_epoch, min_elapsed, train_loss, train_loss))
                sys.stdout.flush()
                losses.append(train_loss)
                print_num = 0
                train_loss = 0.

            # evaluate
            if eval_every > 0 and iter_i % eval_every == 0 or cfg['debug']:

                if cfg['debug']:
                    print("Debugging mode")
                dev_eval = evaluate(model, val_loader, tokenizer=tokenizer, batch_size=eval_batch_size, device=device)
                print("Evaluation Accuracy:", dev_eval['acc'])
                accuracies.append(dev_eval["acc"])


                compare_score = dev_eval["loss"]
                if (compare_score < (best_eval * (1 - cfg["threshold"]))):
                    best_eval = compare_score
                    best_iter = iter_i
                    if not os.path.exists(cfg["save_path"]):
                        os.makedirs(cfg["save_path"])

                    ckpt = {
                        "state_dict": model.state_dict(),
                        "cfg": cfg,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_eval": best_eval,
                        "best_iter": best_iter
                    }    

                    path = os.path.join(cfg["save_path"], "model.pt")
                    torch.save(ckpt, path)
                

            # done training
            if (iter_i + 1) == num_iterations:
                print("# Done training")

                # save last model
                print("# Saving last model")
                ckpt = {
                    "state_dict": model.state_dict(),
                    "cfg": cfg,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_eval": dev_eval['loss'],
                    "best_iter": iter_i
                }
                path = os.path.join(cfg["save_path"], f"last_model.pt")
                torch.save(ckpt, path)

                # evaluate on test with best model
                print("# Loading best model")
                path = os.path.join(cfg["save_path"], f"model.pt")
                if os.path.exists(path):
                    ckpt = torch.load(path)
                    model.load_state_dict(ckpt["state_dict"])
                else:
                    print("No model found.")

                print("# Evaluating")
                dev_eval = evaluate(
                    model, data_loader=val_loader, batch_size=eval_batch_size, tokenizer=tokenizer,
                    device=device)
                test_eval = evaluate(
                    model, data_loader=test_loader, batch_size=eval_batch_size, tokenizer=tokenizer,
                    device=device)

                print("best model iter {:d}: "
                      "dev {} test {}".format(
                    best_iter,
                    make_kv_string(dev_eval),
                    make_kv_string(test_eval)))

                # save result
                result_path = os.path.join(cfg["save_path"], f"results.json")

                cfg["best_iter"] = best_iter

                for k, v in dev_eval.items():
                    cfg["dev_" + k] = v
                    writer.add_scalar('best/dev/' + k, v, iter_i)

                for k, v in test_eval.items():
                    print("test", k, v)
                    cfg["test_" + k] = v
                    writer.add_scalar('best/test/' + k, v, iter_i)

                writer.close()

                with open(result_path, mode="w") as f:
                    json.dump(cfg, f)

                return losses, accuracies


        dev_eval = evaluate(model, val_loader, tokenizer=tokenizer, batch_size=eval_batch_size, device=device)
        print("Evaluation Accuracy:", dev_eval['acc'])
        accuracies.append(dev_eval["acc"])


        compare_score = dev_eval["loss"]
        if (compare_score < (best_eval * (1 - cfg["threshold"]))):
            best_eval = compare_score
            best_iter = iter_i
            if not os.path.exists(cfg["save_path"]):
                os.makedirs(cfg["save_path"])

            ckpt = {
                "state_dict": model.state_dict(),
                "cfg": cfg,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_eval": best_eval,
                "best_iter": best_iter
            }    

            path = os.path.join(cfg["save_path"], "model.pt")
            torch.save(ckpt, path)
                

        epoch += 1
