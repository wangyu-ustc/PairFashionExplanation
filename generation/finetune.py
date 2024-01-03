# copied from the branch zexue
import argparse
import json
import os

import pdb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim
from evaluate import load
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import TensorDataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def bleu_score(references, generated, n_gram=4, smooth=False):
    """a list of lists of tokens"""
    formatted_ref = [[ref] for ref in references]
    # bleu_s, _, _, _, _, _ = compute_bleu(formatted_ref, generated, n_gram, smooth)
    bleu = load("bleu")
    bleu_s = bleu.compute(predictions=generated, references=references, max_order=n_gram, smooth=smooth)['bleu']
    return bleu_s * 100

def set_seed(seed):
    if seed == -1:
        seed = int(torch.randint(0, 2 ** 32 - 1, (1,)).item())

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_tokenized(prompts, labels, tokenizer, batch_size):
    assert len(prompts) == len(labels)

    prompts_input_ids = []
    prompts_attention_masks = []
    labels_input_ids = []
    labels_attention_masks = []

    for i in range(0, len(prompts), batch_size):

        cur_labels_batch = labels[i:i + batch_size]
        cur_prompts_batch = prompts[i: i + batch_size]

        prompts_encoded_dict = tokenizer(
            cur_prompts_batch,
            add_special_tokens=False,
            max_length=100,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        prompts_input_ids.append(prompts_encoded_dict['input_ids'])
        prompts_attention_masks.append(prompts_encoded_dict['attention_mask'])

        if args.model == 'gpt2':
            labels_encoded_dict = tokenizer(
                cur_labels_batch,
                add_special_tokens=False,
                max_length=100,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
        elif "flan-t5" in args.model:

            new_batch = []
            for pmt, txt in zip(cur_prompts_batch, cur_labels_batch):
                new_batch.append(txt[len(pmt):].strip())

            labels_encoded_dict = tokenizer(
                new_batch,
                add_special_tokens=False,
                max_length=100,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',
            )

        labels_input_ids.append(labels_encoded_dict['input_ids'])
        labels_attention_masks.append(labels_encoded_dict['attention_mask'])

    prompts_input_ids = torch.cat(prompts_input_ids, dim=0)
    labels_input_ids = torch.cat(labels_input_ids, dim=0)

    prompts_attention_masks = torch.cat(prompts_attention_masks, dim=0)
    labels_attention_masks = torch.cat(labels_attention_masks, dim=0)

    dataset = TensorDataset(prompts_input_ids, labels_input_ids, prompts_attention_masks, labels_attention_masks)
    return dataset


def compute_generated_ppl(results, labels, prompt_len):
    logits = torch.cat(results[1])
    labels = labels[prompt_len:]
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels[:, len(logits):].view(-1))
    ppl = torch.exp(loss)
    return loss, ppl


def generate(model, inputs, tokenizer):
    result = model.generate(
        **inputs,
        return_dict_in_generate=True,
        output_scores=True,
        max_length=100,
        num_beams=5,
        num_return_sequences=5,
    )

    # for i, output in enumerate(result[0]):
    decoded = tokenizer.decode(result[0][0], skip_special_tokens=True)
    return decoded, result


def extend_instance(obj, cls):
    """Apply mixins to a class instance after creation"""
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(base_cls_name, (base_cls, cls), {})


def evaluation(model, prompts, labels, tokenizer, bleurt, perplexity, device):
    generated_list = []
    ppl_list = []
    loss_list = []
    gt_list = []
    # bleu = evaluate.load("bleu")
    bleu_list = []
    model.eval()
    print("Starting generation and eval... total ", len(prompts))
    with torch.no_grad():

        targets = []

        for idx, pmt in enumerate(prompts):  # 1 by 1
            label = labels[idx]
            inputs = tokenizer(pmt, add_special_tokens=False, return_tensors='pt').to(device)
            generated, result = generate(model, inputs, tokenizer)

            if args.model == 'gpt2':
                generated = generated[len(pmt):]

            generated_list.append(generated)
            label = label[len(pmt):]
            gt_list.append(label)
            targets.append(label)

        if bleurt is not None:
            bleu_list = bleurt.compute(predictions=generated_list, references=targets)['scores']
        else:
            bleu_list = [
                bleu_score(targets, generated_list, n_gram=1, smooth=False),
                bleu_score(targets, generated_list, n_gram=4, smooth=False),
            ]

        for j in range(len(targets)):
            print("reference:", targets[j])
            print("predictions:", generated_list[j])
            if j > 5:
                break

        results = perplexity.compute(model_id='gpt2',
                                     add_start_token=False,
                                     predictions=generated_list)
        ppl_list = np.mean(results['perplexities'])

    print("Generated result:  BLEU {:.6f}, Perplexity: {:.6f}".format(
        np.mean(bleu_list), np.mean(ppl_list)
    ))
    return np.mean(bleu_list), generated_list, gt_list


def train(model,
          tokenizer,
          bleurt,
          perplexity,
          train_loader,
          dev_prompts,
          dev_labels,
          test_prompts,
          test_labels,
          optimizer,
          args,
          device):

    model.train()
    model.to(device)
    best_bleu = -1.0
    loss_fct = CrossEntropyLoss(reduction="none")
    if args.gradient_accumulation_steps > 1:
        accelerator = Accelerator(gradient_accumulation_steps=5)
    else:
        accelerator = Accelerator()

    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for epoch in range(args.epochs):
        total_loss = 0
        total_ppl = 0
        total = 0
        for idx, batch in enumerate(train_loader):

            with accelerator.accumulate(model):

                optimizer.zero_grad()

                prompts, labels, prompts_attention_mask, labels_attention_mask = batch[0].to(device), batch[1].to(device), \
                batch[2].to(device), batch[3].to(device)

                if args.model == 'gpt2':

                    input_ids = labels.clone()
                    labels[torch.where(prompts_attention_mask == 1)] = -100
                    outputs = model(input_ids=input_ids,
                                    attention_mask=labels_attention_mask,
                                    labels=labels)
                    # if evaluate:
                    shift_logits = outputs[1][..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    # Flatten the tokens
                    loss_per_example = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),
                                                       reduction='none')
                    # assert loss_per_example.mean()==loss, "loss not same"
                    ppl = torch.exp(loss_per_example.reshape(shift_logits.shape[0], -1).mean(-1)).sum().item()

                    loss = outputs['loss']
                    # loss = loss_dict[0]
                    # loss.backward()


                elif 'flan-t5' in args.model:

                    # labels[labels == tokenizer.pad_token_id] = -100
                    outputs = model(input_ids=prompts,
                                    attention_mask=prompts_attention_mask,
                                    labels=labels)

                    # if evaluate
                    logits = outputs[1][..., :, :].contiguous()
                    labels = labels.contiguous()
                    # Flatten the tokens
                    loss_per_example = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='none')
                    # assert loss_per_example.mean()==loss, "loss not same"
                    ppl = torch.exp(loss_per_example.reshape(logits.shape[0], -1).mean(-1)).sum().item()

                    loss = outputs['loss']
                    # loss.backward()

                accelerator.backward(loss)
                optimizer.step()

                # # =====================
                # shift_logits = loss_dict[1][..., :-1, :].contiguous()
                # shift_labels = labels[..., 1:].contiguous()
                # # Flatten the tokens
                # loss_per_example = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                # # assert loss_per_example.mean()==loss, "loss not same"
                # perplexity = torch.exp(loss_per_example.reshape(shift_logits.shape[0], -1).mean(-1)).sum().item()
                total_ppl += ppl
                total += len(batch[0])
                # =====================
                total_loss += (loss.item() * len(batch[0]))
                if (idx + 1) % args.log_interval == 0:
                    print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, TF PPL: {:.6f}".format(
                            epoch + 1,
                            total, len(train_loader.dataset),
                            100 * total / len(train_loader.dataset), total_loss / total, total_ppl / total
                        )
                    )
                    print("*** Still in training, start generation with dev sampled prompts...")
                    dev_bleu, generated_list, gt_list = evaluation(model, dev_prompts[args.start:args.end],
                                                                   dev_labels[args.start:args.end], tokenizer, bleurt,
                                                                   perplexity, device)
                    if dev_bleu > best_bleu:
                        best_bleu = dev_bleu
                        torch.save(model, os.path.join(args.save_dir, f'{args.model}_model.pt'))
                        print("Save best model at Epoch ", epoch)
                    # print("# Start generation with dev prompts...")
                    # test_bleu=eval(model, test_prompts, test_labels, tokenizer)
                    model.train()
            # print("### Finish epoch", epoch)
            if (epoch + 1) % 10 == 0:
                print("### Epoch: [{}], loss: {:.6f}, ppl:{:.6f}".format(epoch, total_loss / total, total_ppl / total))
                print("### Start generation with sampled dev prompts...")
                dev_bleu, generated_list, gt_list = evaluation(model, dev_prompts[args.start:args.end],
                                                               dev_labels[args.start:args.end], tokenizer, bleurt,
                                                               perplexity,
                                                               device)
                if dev_bleu > best_bleu:
                    best_bleu = dev_bleu
                    torch.save(model, os.path.join(args.save_dir, f'{args.model}_model.pt'))
                    print("Save best model at Epoch ", epoch)

                torch.save(model, os.path.join(args.save_dir, f'{args.model}_last_model.pt'))
                print("Save last model at Epoch ", epoch)

                model.train()

    # ==============


def test(args, model, test_prompts, test_labels, tokenizer, bleurt, perplexity, device):
    print("### Start generation with dev prompts...")
    model = torch.load(os.path.join(args.save_dir, f'{args.model}_model.pt'))
    model.eval()
    test_bleu, generated_list, gt_list = evaluation(model, test_prompts, test_labels, tokenizer, bleurt, perplexity,
                                                    device)
    with open(os.path.join(args.save_dir, "generated.json"), "w") as f:
        json.dump(generated_list, f)
    with open(os.path.join(args.save_dir, "gt.json"), "w") as f:
        json.dump(gt_list, f)


def rephrase(model, items, features):
    prompt = ''
    for idx, (item, feature) in enumerate(zip(items, features)):
        prompt += feature + ' ' + item
        if idx == len(items) - 1:
            pass
        elif idx == len(items) - 2:
            prompt += ' and '
        elif idx < len(items) - 2:
            prompt += ', '

    if model == 'gpt2':
        return prompt + ' match because'
    elif 'flan-t5' in model:
        return "Generate the reason why " + prompt + " match:"


def extract_prompts_and_labels(model, data):
    labels = []
    prompts = []
    for seq, items, features in zip(data['explanations'], data['items'], data['features']):
        items = eval(items)
        features = eval(features)
        prompt = rephrase(model, items, features)
        prompts.append(prompt)
        if model == 'gpt2':
            labels.append(prompt + " " + seq[0].lower() + seq[1:])
        else:
            labels.append(prompt + " " + seq[0].upper() + seq[1:])

    return prompts, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=100, help='How many epochs to run in total?')
    parser.add_argument('-start', '--start', type=int, default=0,
                        help='calculate sampled bleu score start from where (for choosing best model)')
    parser.add_argument('-end', '--end', type=int, default=100,
                        help='calculate sampled bleu score end from where (for choosing best model)')
    parser.add_argument('-l', '--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('-b', '--batch_size', type=int, default=2, help='Batch size during training per GPU')
    parser.add_argument('-log', '--log_interval', type=int, default=5000, help='log iter')
    parser.add_argument('-mode', '--mode', type=str, default='train')
    parser.add_argument('-save', '--save_dir', type=str, default=None, help='log dir')
    parser.add_argument("--model", type=str, default='flan-t5-large', help='which model to use for finetuning')
    parser.add_argument('--lora', default=False, action='store_true')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    args = parser.parse_args()
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))

    print("Loading Bleurt and Perplexity")
    bleurt = load('bleurt', module_type='metric')
    print("bleurt loaded")
    perplexity = load('perplexity', module_type='metric')
    print("perplexity loaded")

    if args.save_dir is None:
        args.save_dir = f'./ckpt/finetune_{args.model}/'

    # data_dir = args.data_dir
    batch_size = args.batch_size
    path = args.save_dir
    if args.model == 'gpt2':
        model = GPT2LMHeadModel.from_pretrained(
            "gpt2",
            output_hidden_states=True
        )
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    elif 'flan-t5' in args.model:
        model = AutoModelForSeq2SeqLM.from_pretrained(f"google/{args.model}")
        tokenizer = AutoTokenizer.from_pretrained(f"google/{args.model}")

        if args.lora:
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()


    else:
        raise NotImplementedError



    optimizer = AdamW(model.parameters(), lr=args.lr)

    csv_file = pd.read_csv("../data/data.csv")
    train_data_raw, test_data_raw = train_test_split(csv_file, test_size=0.1, random_state=0)

    train_data = extract_prompts_and_labels(args.model, train_data_raw)
    test_data = dev_data = extract_prompts_and_labels(args.model, test_data_raw)

    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    # normal:
    train_labels = train_data[1]
    test_labels = test_data[1]

    # for debug:
    # train_labels = [prompt + " " + prompt.replace("match because", "").strip() for prompt in train_data[0]]
    # test_labels = [prompt + " " + prompt.replace("match because", "").strip() for prompt in test_data[0]]

    if args.mode == "train":
        # normal:
        dev_labels = dev_data[1]
        # for debug:
        # dev_labels = [prompt + " " + prompt.replace("match because", "").strip() for prompt in dev_data[0]]

        train_dataset = get_tokenized(train_data[0], train_labels, tokenizer, batch_size)

        train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=batch_size
        )

        # model = torch.load(os.path.join(args.save_dir, f'{args.model}_model.pt'))

        train(model, tokenizer, bleurt, perplexity, train_dataloader,
              dev_data[0], dev_labels,
              test_data[0], test_labels,
              optimizer,
              args,
              device)

    print("Testing set")
    test(args, model, test_data[0], test_labels, tokenizer, bleurt, perplexity, device)

