# from bleu import compute_bleu
import json
import numpy as np
import os
import re
import sys
import pdb
import torch
import pandas as pd
import datetime
import argparse
from tqdm import tqdm
from module import DiscretePromptLearning, ContinuousPromptLearning
from finetune import evaluation, extract_prompts_and_labels
from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, GPT2Config
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from evaluate import load


def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '


def bleu_score(references, generated, n_gram=4, smooth=False):
    """a list of lists of tokens"""
    formatted_ref = [[ref] for ref in references]
    # bleu_s, _, _, _, _, _ = compute_bleu(formatted_ref, generated, n_gram, smooth)
    bleu = load("bleu")
    bleu_s = bleu.compute(predictions=generated, references=references, max_order=n_gram, smooth=smooth)['bleu']
    return bleu_s * 100


def rephrase(items, features):
    prompt = ''
    for idx, (item, feature) in enumerate(zip(items, features)):
        prompt += feature + ' ' + item
        if idx == len(items) - 1:
            pass
        elif idx == len(items) - 2:
            prompt += ' and '
        elif idx < len(items) - 2:
            prompt += ', '
    return prompt + ' match becase'


class GenerationDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, bos, eos):
        self.seq = []
        self.prompt = []
        print("Start preprocessing")

        for seq, items, features in zip(data['explanations'], data['items'], data['features']):
            items = eval(items)
            features = eval(features)

            self.seq.append(seq)
            self.prompt.append(rephrase(items, features))

        print(f"{len(self.seq)} sequences in total")
        print(f"{len(self.prompt)} prompts in total")
        sys.stdout.flush()

        self.data = data
        t = [
            '{} {} {}'.format(bos, x, eos) for x in self.seq
        ]

        encoded_inputs = tokenizer(t, padding=True, return_tensors='pt')
        self.seq = encoded_inputs['input_ids'].contiguous()
        self.mask = encoded_inputs['attention_mask'].contiguous()

        # self.words = data['words'].values
        # self.nouns = data['nouns'].values
        # self.prompt = [' '.join(eval(noun) + eval(word)) for (noun, word) in zip(self.nouns, self.words)]
        encoded_features = tokenizer(self.prompt, padding=True, return_tensors='pt')
        self.prompt = encoded_features['input_ids'].contiguous()
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        return self.seq[idx], self.mask[idx], self.prompt[idx]

    def __len__(self):
        return len(self.seq)


def postprocessing(string):
    '''
    adopted from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    '''
    string = re.sub('\'s', ' \'s', string)
    string = re.sub('\'m', ' \'m', string)
    string = re.sub('\'ve', ' \'ve', string)
    string = re.sub('n\'t', ' n\'t', string)
    string = re.sub('\'re', ' \'re', string)
    string = re.sub('\'d', ' \'d', string)
    string = re.sub('\'ll', ' \'ll', string)
    string = re.sub('\(', ' ( ', string)
    string = re.sub('\)', ' ) ', string)
    string = re.sub(',+', ' , ', string)
    string = re.sub(':+', ' , ', string)
    string = re.sub(';+', ' . ', string)
    string = re.sub('\.+', ' . ', string)
    string = re.sub('!+', ' ! ', string)
    string = re.sub('\?+', ' ? ', string)
    string = re.sub(' +', ' ', string).strip()
    return string


def ids2tokens(ids, tokenizer, eos):
    text = tokenizer.decode(ids, skip_special_tokens=True)
    text = postprocessing(text)  # process punctuations: "good!" -> "good !"
    tokens = []
    for token in text.split():
        if token == eos:
            break
        tokens.append(token)
    return ' '.join(tokens)


def calculate_ppl():
    csv_file = pd.read_csv("../data/statistics_item_feature_filtered.csv")
    train_data_raw, test_data_raw = train_test_split(csv_file, test_size=0.1, random_state=0)
    test_data = dev_data = extract_prompts_and_labels(args.model, test_data_raw)
    targets = test_data[1]
    prompts = test_data[0]


def generate(args):
    csv_file = pd.read_csv("../data/statistics_item_feature_filtered.csv")
    train_data_raw, test_data_raw = train_test_split(csv_file, test_size=0.1, random_state=0)

    if args.model == 'naive':
        test_data = dev_data = extract_prompts_and_labels('gpt2', test_data_raw)
        targets = test_data[1]
        prompts = test_data[0]
        print(f"{len(prompts)} in total")

        new_targets = []
        for txt in targets:
            words = txt.split(" match because ")
            if len(words) < 2:
                print(txt)
                pdb.set_trace()
            new_targets.append(words[1])
        targets = new_targets

        generated = []
        for pmt in prompts:
            generated.append(pmt.replace("match because", "are a good match"))

        return generated, targets

    if args.model == 'chatgpt':
        test_data = dev_data = extract_prompts_and_labels('gpt2', test_data_raw)
        targets = test_data[1]
        prompts = test_data[0]
        print(f"{len(prompts)} in total")

        new_targets = []
        for txt in targets:
            words = txt.split(" match because ")
            if len(words) < 2:
                print(txt)
                pdb.set_trace()
            new_targets.append(words[1])
        targets = new_targets

        generated = []

        results = json.load(open("../results/responses_for_stage2_simplified.json", "r"))
        for result in results:
            generated.append(result['response'])

        return generated, targets

    if args.model == 'gpt2':
        test_data = dev_data = extract_prompts_and_labels(args.model, test_data_raw)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained(
            "gpt2",
            output_hidden_states=True
        )
        if args.finetuned:
            # model = torch.load(os.path.join(args.save_dir, f"{args.model}_model.pt"))
            model = torch.load(f"./ckpt/{args.ckpt_dir}/{args.model}_model.pt")
        model = model.cuda()
        model.eval()

        generated_list = []
        targets = test_data[1]
        prompts = test_data[0]
        print(f"{len(prompts)} in total")

        new_targets = []
        for txt in targets:
            words = txt.split(" match because ")
            if len(words) < 2:
                print(txt)
                pdb.set_trace()
            new_targets.append(words[1])
        targets = new_targets

        for idx, pmt in tqdm(enumerate(prompts)):
            inputs = tokenizer(pmt, add_special_tokens=False, return_tensors='pt').to('cuda')
            result = model.generate(
                **inputs,
                return_dict_in_generate=True,
                output_scores=True,
                max_length=100,
                repetition_penalty=2.0,
                num_beams=5,
                num_return_sequences=5,
            )
            generated = tokenizer.decode(result[0][0], skip_special_tokens=True)
            generated_list.append(generated)

        # test_bleu, generated_list, gt_list = evaluation(model, test_data[0], test_data[1], tokenizer, bleurt, perplexity,
        # device)

        new_gts = []
        for txt in generated_list:
            words = txt.split(" match because")
            if len(words) < 2:
                print(txt)
                pdb.set_trace()
            new_gts.append(words[1].strip())
        generated_list = new_gts

        # targets = [txt.split(" match because ")[1] for txt in targets]

        return generated_list, targets


    elif 'flan-t5' in args.model:
        test_data = dev_data = extract_prompts_and_labels(args.model, test_data_raw)
        model = AutoModelForSeq2SeqLM.from_pretrained(f"google/{args.model}")
        tokenizer = AutoTokenizer.from_pretrained(f"google/{args.model}")

        if args.finetuned:
            # model = torch.load(f"./ckpt/finetune_{args.model}/{args.model}_model.pt")
            model = torch.load(f"./ckpt/{args.ckpt_dir}/{args.model}_last_model.pt")

        model = model.cuda()
        model.eval()

        generated_list = []
        targets = test_data[1]
        prompts = test_data[0]
        print(f"{len(prompts)} in total")

        new_targets = []
        for pmt, txt in tqdm(zip(prompts, targets)):
            new_targets.append(txt[len(pmt):])
            # words = txt.split(" match because ")
            # if len(words) < 2:
            #     print(txt)
            #     pdb.set_trace()
            # new_targets.append(words[1])
        targets = new_targets

        generated_list = []
        for idx, pmt in tqdm(enumerate(prompts)):
            if idx == 0:
                print("Example of prompt:", pmt)

            inputs = tokenizer(pmt, return_tensors="pt")
            inputs['input_ids'] = inputs['input_ids'].cuda()
            inputs['attention_mask'] = inputs['attention_mask'].cuda()
            outputs = model.generate(**inputs,
                                     return_dict_in_generate=True,
                                     output_scores=True,
                                     max_length=100,
                                     num_beams=5,
                                     num_return_sequences=5,
                                     repetition_penalty=2.0)
            # pdb.set_trace()
            # logits_all = outputs[2]
            # tokens = outputs[0][0].clone()
            # tokens[torch.where(tokens==0)] = -100
            # ppl = 1

            # logits = []

            # for i in range(len(logits_all)):
            #     # logits[i]: i-th step
            #     # logits[i][0]: beam search first result logits
            #     # logits[i][0][tokens[0][i+1]] get the logit
            #     # ppl *= logits[i][0][tokens[0][i+1]]
            #     logits.append(logits_all[i][0])

            # logits = torch.stack(logits)

            # losses = F.cross_entropy(logits[:-1], tokens[1:], reduction='none')
            # losses = losses[torch.where(losses > 0)]
            # ppl = torch.exp(losses.mean())

            generated = tokenizer.decode(outputs[0][0], skip_special_tokens=True)

            # generated = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)[0]
            generated_list.append(generated)

        return generated_list, targets


    elif args.model == 'pepler':
        device = 'cuda'
        bos = '<bos>'
        eos = '<eos>'
        pad = '<pad>'
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token=bos, eos_token=eos, pad_token=pad)
        test_dataset = GenerationDataset(test_data_raw, tokenizer, bos, eos)
        test_data = torch.utils.data.DataLoader(
            test_dataset, shuffle=False, batch_size=args.batch_size
        )
        model = torch.load("./ckpt/pepler_model.pt")
        model.to('cuda')
        ids_test = torch.cat([ids for ids, _, _ in test_data])
        references = [ids2tokens(ids[1:], tokenizer, eos).strip() for ids in ids_test]
        references = [ref[0].lower() + ref[1:] for ref in references]

        idss_predict = []
        with torch.no_grad():
            for seq, _, prompt in test_data:
                prompt = prompt.to(device)
                text = seq[:, :1].to(device)  # bos, (batch_size, 1)
                # pdb.set_trace()
                for idx in range(seq.size(1)):
                    # produce a word at each step
                    outputs = model(prompt, text, None)
                    last_token = outputs.logits[:, -1, :]  # the last token, (batch_size, ntoken)
                    word_prob = torch.softmax(last_token, dim=-1)
                    token = torch.argmax(word_prob, dim=1,
                                         keepdim=True)  # (batch_size, 1), pick the one with the largest probability
                    text = torch.cat([text, token], 1)  # (batch_size, len++)
                ids = text[:, 1:].tolist()  # remove bos, (batch_size, seq_len)
                idss_predict.extend(ids)

        tokens_predict = [ids2tokens(ids, tokenizer, eos).strip() for ids in idss_predict]
        return tokens_predict, references


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=10, help='How many epochs to run in total?')
    parser.add_argument('-start', '--start', type=int, default=123,
                        help='calculate sampled bleu score start from where (for chosing best model)')
    parser.add_argument('-end', '--end', type=int, default=223,
                        help='calculate sampled bleu score end from where (for chosing best model)')
    parser.add_argument('-l', '--lr', type=float, default=2.5e-5, help='learning rate')
    parser.add_argument('-b', '--batch_size', type=int, default=6, help='Batch size during training per GPU')
    parser.add_argument('-log', '--log_interval', type=int, default=200, help='log iter')
    parser.add_argument('-data', '--data_dir', type=str, default='/data2/zexue/debias_by_rationale/data/bold_all/',
                        help='log dir')
    parser.add_argument('-mode', '--mode', type=str, default='train')
    parser.add_argument("--model", type=str, default='gpt2')
    parser.add_argument("--ckpt_dir", type=str, default='gpt2-rp2')
    parser.add_argument('-save', '--save_dir', type=str, default='./ckpt/gpt2-rp2/', help='log dir')

    parser.add_argument("--finetuned", action='store_true', default=False)
    args = parser.parse_args()

    model_string = args.model + ("_" + args.ckpt_dir + '_finetuned' if args.finetuned else "")
    if os.path.exists(f"./stage2_v1/{model_string}_generated.json"):
        generated = json.load(open(f"./stage2_v1/{model_string}_generated.json", 'r'))
        gt = json.load(open(f"./stage2_v1/{model_string}_gt.json", 'r'))
    else:
        print(now_time() + 'Starting Generating...')
        generated, gt = generate(args)

        with open(os.path.join(f"./stage2_v1/{model_string}_generated.json"), "w") as f:
            json.dump(generated, f)
        f.close()
        with open(os.path.join(f"./stage2_v1/{model_string}_gt.json"), "w") as f:
            json.dump(gt, f)
        f.close()

    rouge = load("rouge")
    # bleurt = load('bleurt', module_type='metric')
    bleurt = load('bleurt', 'bleurt-large-512', module_type='metric')
    perplexity = load('perplexity', module_type='metric')
    # bleu_list = bleurt.compute({
    #     'predictions': generated,
    #     'references': gt
    # })['scores']
    bleu_list = bleurt.compute(predictions=generated, references=gt)['scores']
    rouge_list = rouge.compute(predictions=generated, references=gt)

    print(rouge_list)
    lengths = np.array([len(txt) for txt in generated])
    if len(np.where(lengths == 0)[0]) > 0:
        print("Empty sentences:", len(np.where(lengths)[0]))

    gt = np.array(gt)[np.where(lengths > 0)].tolist()
    generated = np.array(generated)[np.where(lengths > 0)].tolist()

    ppl_list = np.mean(perplexity.compute(model_id='gpt2',
                                          add_start_token=True,
                                          predictions=generated)['perplexities'])
    print("Generated result:  BLEU {:.6f}, Perplexity: {:.6f}".format(
        np.mean(bleu_list), np.mean(ppl_list)
    ))

    BLEU1 = bleu_score(gt, generated, n_gram=1, smooth=False)
    print(now_time() + 'BLEU-1 {:7.4f}'.format(BLEU1))
    BLEU4 = bleu_score(gt, generated, n_gram=4, smooth=False)
    print(now_time() + 'BLEU-4 {:7.4f}'.format(BLEU4))







