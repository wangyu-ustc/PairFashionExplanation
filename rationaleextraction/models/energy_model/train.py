import os
import time
import numpy as np
import json

import torch
import torch.optim
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import BertTokenizer, GPT2Tokenizer
from models.common.util import make_kv_string
from models.energy_model.models.model_helpers import build_model
from models.energy_model.util import prepare_minibatch, print_parameters, \
    initialize_model_, get_device
from models.energy_model.evaluate import evaluate
from models.energy_model.dataset import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

device = get_device()
print("device:", device)

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


def train(cfg):
    """
    Main training loop.
    """

    cfg = vars(cfg)

    set_seed(cfg['seed'])

    print(cfg)

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

    print("Loading data")
    dataset = Dataset(
        path=cfg['path'],
        tokenizer=tokenizer,
        debug=cfg['debug'],
        concat=True
    )

    train_idxes, valid_idxes, test_idxes = train_valid_test_split(np.arange(len(dataset)), 0.8, 0.1, 0.1)
    # num_workers = multiprocessing.cpu_count() if 
    num_workers = 0 if cfg['num_workers'] == -1 else cfg['num_workers']

    train_loader = DataLoader(Subset(dataset, indices=train_idxes), batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=num_workers)

    val_loader = DataLoader(Subset(dataset, indices=valid_idxes), batch_size=cfg['batch_size'], shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(Subset(dataset, indices=test_idxes), batch_size=cfg['batch_size'],
                             shuffle=False, num_workers=num_workers)

    print("train", len(train_idxes))
    print("dev", len(valid_idxes))
    print("test", len(test_idxes))

    iters_per_epoch = len(train_loader)

    if cfg["eval_every"] == -1:
        eval_every = iters_per_epoch
        print("Set eval_every to {}".format(iters_per_epoch))

    if cfg["num_iterations"] < 0:
        num_iterations = iters_per_epoch * -1 * cfg["num_iterations"]
        print("Set num_iterations to {}".format(num_iterations))

    model = build_model(cfg)
    initialize_model_(model)

    # Build model
    optimizer = Adam(model.parameters(), lr=cfg["lr"],
                     weight_decay=cfg["weight_decay"])

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg["lr_decay"], patience=cfg["patience"],
        verbose=True, cooldown=cfg["cooldown"], threshold=cfg["threshold"],
        min_lr=cfg["min_lr"])

    iter_i = 0
    train_loss = 0.
    print_num = 0
    start = time.time()
    losses = []
    accuracies = []
    best_eval = 1.0e9
    best_iter = 0
    epoch = 0

    model = model.to(device)

    if cfg['pretrained_weights'] is not None:
        print("Loading pretrained weights from %s" % cfg['pretrained_weights'])
        ckpt = torch.load(cfg['pretrained_weights'])
        model.load_state_dict(ckpt['state_dict'])

    # print model
    print(model)
    print_parameters(model)

    while True:  # when we run out of examples, shuffle and continue

        for idx, batch in enumerate(train_loader):

            model.train()

            x, targets = batch
            prepared_batch = prepare_minibatch(x, targets, device, return_reverse_map=True, concat=True)

            if prepared_batch is None: continue

            x, mask, targets, reverse_map = prepared_batch

            logits, _, _ = model(x)  # forward pass

            loss, loss_optional = model.get_loss(logits, targets, mask=mask)

            model.zero_grad()  # erase previous gradients

            train_loss += loss.item()
            loss.backward()

            optimizer.step()

            print_num += 1
            iter_i += 1

            # print info
            if iter_i % print_every == 0:
                train_loss = train_loss / print_every

                print_str = make_kv_string(loss_optional)
                min_elapsed = (time.time() - start) // 60
                print("Epoch %r Iter %r time=%dm loss=%.4f %s" %
                      (epoch, iter_i, min_elapsed, train_loss, print_str))
                losses.append(train_loss)
                print_num = 0
                train_loss = 0.

            # evaluate
            if iter_i % eval_every == 0:
                dev_eval = evaluate(model, val_loader, tokenizer=tokenizer,
                                           batch_size=eval_batch_size, device=device)
                accuracies.append(dev_eval["acc"])

                print("# epoch %r iter %r: dev %s \n\n" % (
                    epoch, iter_i, make_kv_string(dev_eval)))

                # save best model parameters
                compare_score = dev_eval["loss"]
                if "obj" in dev_eval:
                    compare_score = dev_eval["obj"]

                scheduler.step(compare_score)  # adjust learning rate

                if (compare_score < (best_eval * (1 - cfg["threshold"]))) and \
                        iter_i > (3 * iters_per_epoch):
                    print("***highscore*** %.4f" % compare_score)
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
                    path = os.path.join(cfg["save_path"], f"model.pt")
                    torch.save(ckpt, path)

            # done training
            if iter_i == num_iterations:
                print("# Done training")

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
                    model, val_loader, tokenizer, batch_size=eval_batch_size,
                    device=device)
                test_eval = evaluate(
                    model, test_loader, tokenizer, batch_size=eval_batch_size,
                    device=device)

                print("best model iter {:d}: "
                      "dev {} test {}".format(
                    best_iter,
                    make_kv_string(dev_eval),
                    make_kv_string(test_eval)))

                # save result
                result_path = os.path.join(cfg["save_path"], f"results.json")

                cfg["best_iter"] = best_iter

                with open(result_path, mode="w") as f:
                    json.dump(cfg, f)

                return losses, accuracies

        epoch += 1

