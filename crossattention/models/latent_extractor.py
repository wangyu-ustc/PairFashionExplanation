#!/usr/bin/env python
import pdb

import torch
import copy
from torch import nn
import numpy as np

from typing import Union

from transformers import GPT2Model, BertModel
from torch.nn.functional import softplus, sigmoid, tanh

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_VAL = 1e6


__all__ = ["CrossAttentionModel"]


class CrossAttentionModel(nn.Module):
    
    def __init__(self,
                 cfg,
                 # vocab:          object = None,
                 # vocab_size:     int = 0,
                 output_size:    int = 1,
                 selection:      float = 1.0,
                 lasso:          float = 0.0,
                 llm:           str = 'bert',
                 ):

        super(CrossAttentionModel, self).__init__()

        self.cfg = cfg
        self.output_size = output_size
        self.selection = selection
        self.lasso = lasso

        if llm == 'bert':
            model = BertModel.from_pretrained("bert-base-uncased")
            self.embed = copy.deepcopy(model.embeddings.word_embeddings)
        else:
            model = GPT2Model.from_pretrained("gpt2")
            self.embed = copy.deepcopy(model.wte)

        self.trans = nn.Sequential(
            nn.Linear(2 * self.embed.weight.shape[1], 1024),
            nn.ReLU(),
            nn.Linear(1024, self.embed.weight.shape[1])
        )
        
        self.attn = nn.Sequential(
            nn.Linear(self.embed.weight.shape[1], self.embed.weight.shape[1]),
            nn.ReLU(),
            nn.Linear(self.embed.weight.shape[1], 1)
        )

        self.classifier = nn.Linear(self.embed.weight.shape[1], output_size)
        self.criterion = nn.CrossEntropyLoss(reduction='none') # classification

    def predict(self, logits, **kwargs):
        """
        Predict deterministically.
        :param x:
        :return: predictions, optional (dict with optional statistics)
        """
        assert not self.training, "should be in eval mode for prediction"
        return logits.argmax(-1)


    def cross_attn(self, x1, x2, mask1, mask2, z1=None, z2=None):

        l1 = x1.size(dim=1)
        l2 = x2.size(dim=1)

        # 50257 is a number both suitable for bert and gpt2
        x1[torch.where(x1==50257)] = 0
        x2[torch.where(x2==50257)] = 0

        # x1: [b, l1], x2: [b, l2]
        with torch.no_grad():
            diff = x1.unsqueeze(1) - x2.unsqueeze(-1) # [b, l2, l1]
            diff_mask = diff != 0

        emb1 = self.embed(x1)
        emb2 = self.embed(x2)

        # Now emb1 and emb2: [b, l, e]
        emb1 = emb1.unsqueeze(dim=1).repeat(1, l2, 1, 1) # (b, l2, l1, e)
        emb2 = emb2.unsqueeze(dim=2).repeat(1, 1, l1, 1) # (b, l2, l1, e)

        if z1 is not None:
            mask1 = (z1 * mask1.float()).unsqueeze(dim=1).repeat(1, l2, 1) # (b, l2, l1)
            mask2 = (z2 * mask2.float()).unsqueeze(dim=2).repeat(1, 1, l1) # (b, l2, l1)
        else:
            mask1 = (mask1.float()).unsqueeze(dim=1).repeat(1, l2, 1) # (b, l2, l1)
            mask2 = (mask2.float()).unsqueeze(dim=2).repeat(1, 1, l1) # (b, l2, l1)

        embed = self.trans(torch.cat([emb1, emb2], dim=-1)) # [b, l2, l1, e]

        #### Cross-Attention
        attn_logits = self.attn(embed)
        attn_logits.masked_fill_(((mask1 == 0) | (mask2 == 0) | (diff_mask == 0)).unsqueeze(dim=-1), -MAX_VAL)
        
        attn_weights = torch.sigmoid(attn_logits)

        if self.lasso > 0:
            lasso_penalty = torch.mean(torch.abs(attn_weights)) * self.lasso
        else:
            lasso_penalty = 0

        embed = (attn_weights / (torch.sum(attn_weights, dim=(1, 2, 3), keepdim=True) + 1e-5) * embed).sum(dim=1).sum(dim=1)  # (b, 2*e)
        # embed = (attn_weights * embed).sum(dim=1).sum(dim=1)

        return embed, attn_weights, lasso_penalty

    def fill_cat(self, x):
        
        max_length = np.max([len(x_i) for x_i in x])
        new_x = []
        for x_i in x:
            try:
                new_x.append(torch.cat([x_i, torch.tensor([0] * (max_length - len(x_i))).to(x_i.device)]))
            except:
                new_x.append(torch.cat([x_i.long(), torch.tensor([0] * (max_length - len(x_i))).to(x_i.device)]))

        return torch.stack(new_x)

    def forward(self, x, mask,
                    reverse_map=None, 
                    return_attn=False, 
                    return_lasso=False,
                    return_mask=False):
        """
        Generate a sequence of zs with the Generator.
        Then predict with sentence x (zeroed out with z) using Encoder.

        :param x: [B, T] (that is, batch-major is assumed)
        :return:
        """
        
        x1, x2 = x
        mask1, mask2 = mask
        if reverse_map is not None:
            reverse_map1, reverse_map2 = reverse_map

        if mask1 is None:
            mask1 = (x1 != 50257)  # [B,T]
        if mask2 is None:
            mask2 = (x2 != 50257)

        if reverse_map is not None:
            x1, mask1 = x1[reverse_map1], mask1[reverse_map1]
            x2, mask2 = x2[reverse_map2], mask2[reverse_map2]
            
        embed, attn, lasso_penalty = self.cross_attn(x1, x2, mask1, mask2)

        y = self.classifier(embed)

        results = [y]

        if return_attn:
            results += [attn]
        
        if return_lasso:
            results += [lasso_penalty]
        
        return tuple(results)


    def get_loss(self, logits, targets, mask=None, **kwargs):

        return self.criterion(logits, targets).mean()