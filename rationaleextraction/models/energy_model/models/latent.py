#!/usr/bin/env python

import pdb
import copy
import torch
from torch import nn
import numpy as np

from transformers import GPT2LMHeadModel, BertModel
from models.common.util import get_z_stats
from models.common.classifier import Classifier

from models.common.latent import EPS, DependentLatentModel
from torch.nn.functional import softplus, sigmoid, tanh

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


__all__ = ["LatentRationaleModel"]


class LatentRationaleModel(nn.Module):
    """
    Latent Rationale
    Categorical output version (for SST)

    Consists of:

    p(y | x, z)     observation model / classifier
    p(z | x)        latent model

    """
    def __init__(self,
                 cfg,
                 emb_size:       int = 200,
                 hidden_size:    int = 200,
                 output_size:    int = 1,
                 dropout:        float = 0.1,
                 layer:          str = "lstm",
                 dependent_z:    bool = False,
                 z_rnn_size:     int = 30,
                 selection:      float = 1.0,
                 lasso:          float = 0.0,
                 lambda_init:    float = 1e-4,
                 lagrange_lr:    float = 0.01,
                 lagrange_alpha: float = 0.99,
                 strategy:       int = 0,
                 ):

        super(LatentRationaleModel, self).__init__()

        self.cfg = cfg
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.selection = selection
        self.lasso = lasso
        self.layer = layer

        self.lagrange_alpha = lagrange_alpha
        self.lagrange_lr = lagrange_lr
        self.lambda_init = lambda_init
        self.z_rnn_size = z_rnn_size
        self.dependent_z = dependent_z

        # self.embed = embed = nn.Embedding(vocab_size, emb_size, padding_idx=1)

        model = BertModel.from_pretrained("bert-base-uncased")
        self.embed = embed = copy.deepcopy(model.embeddings.word_embeddings)

        if layer == 'rcnn' or layer == 'lstm':
            self.classifier = Classifier(
            embed=embed, hidden_size=hidden_size, output_size=output_size,
            dropout=dropout, layer=layer, nonlinearity="softmax")
        elif layer == 'bert':
            self.classifier = nn.Sequential(
                BertModel.from_pretrained('bert-base-cased'),
                nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(768, output_size),
                    nn.LogSoftmax(dim=-1)
                )
            )
        else:
            raise NotImplementedError
        

        self.latent_model = DependentLatentModel(
            embed=embed, hidden_size=hidden_size,
            dropout=dropout, layer='lstm', strategy=strategy)

        self.criterion = nn.NLLLoss(reduction='none') # classification

        # lagrange
        self.lagrange_alpha = lagrange_alpha
        self.lagrange_lr = lagrange_lr
        self.lambda_min = self.lambda_init / 10
        self.register_buffer('lambda0', torch.full((1,), lambda_init))
        self.register_buffer('lambda1', torch.full((1,), lambda_init))
        self.register_buffer('c0_ma', torch.full((1,), 0.))  # moving average
        self.register_buffer('c1_ma', torch.full((1,), 0.))  # moving average
        self.strategy = strategy

    @property
    def z(self):
        return self.latent_model.z

    @property
    def z_layer(self):
        return self.latent_model.z_layer

    @property
    def z_dists(self):
        return self.latent_model.z_dists

    def predict(self, logits, **kwargs):
        """
        Predict deterministically.
        :param x:
        :return: predictions, optional (dict with optional statistics)
        """
        assert not self.training, "should be in eval mode for prediction"
        return logits.argmax(-1)


    # def forward(self, x):
    #     mask = (x != 50257)
    #     x[torch.where(x==50257)] = 0
    #     assert (x < 50257).all()
    #     pdb.set_trace()
    #     out = self.classifier[0](input_ids=x, attention_mask=mask)
    #     pdb.set_trace()
    #     y = self.classifier[1](out[1])
    #     z = (torch.ones(x.shape) * mask).bool()
    #     return y, z, z


    def forward(self, x):
        """
        Generate a sequence of zs with the Generator.
        Then predict with sentence x (zeroed out with z) using Encoder.

        :param x: [B, T] (that is, batch-major is assumed)
        :return:
        """
        mask = (x != 50257)  # [B,T]
        x[torch.where(x==50257)] = 0
        z = self.latent_model(x, mask)
        # z = torch.ones(x.shape).to(x.device)

        emb = self.embed(x)
        # apply z to main inputs
        if z is not None:
            z_mask = (mask.float() * z).unsqueeze(-1)  # [B, T, 1]
            rnn_mask = z_mask.squeeze(-1) > 0.  # z could be continuous
            emb = emb * z_mask
        
        if self.layer == 'bert':
            out = self.classifier[0](inputs_embeds=emb, attention_mask=mask)
            y = self.classifier[1](out[1])
        else:
            y = self.classifier(x, mask, z)
    
        return y, z, z #! need to change it to y, z, logits

    def get_loss(self, logits, targets, mask=None, **kwargs):

        optional = {}
        selection = self.selection
        lasso = self.lasso

        loss_vec = self.criterion(logits, targets)  # [B]

        # reweight:
        # weights = torch.tensor([66319, 53252,  3929,  1571])[targets]
        # weights = 1 / weights * 66319
        # loss_vec = loss_vec * weights.to(loss_vec.device)

        # main MSE loss for p(y | x,z)
        ce = loss_vec.mean()        # [1]
        optional["ce"] = ce.item()  # [1]


        batch_size = mask.size(0)
        lengths = mask.sum(1).float()  # [B]

        # z = self.generator.z.squeeze()
        z_dists = self.latent_model.z_dists

        # pre-compute for regularizers: pdf(0.)
        if len(z_dists) == 1:
            # pdf0 = z_dists[0].pdf(0.)
            cdf_0_5 = z_dists[0].cdf(0.5)
            raise NotImplementedError
        else:
            if self.strategy == 2:
                # Yu's Strategy
                cdf_0_5 = []
                for t in range(len(z_dists)):
                    cdf_t = z_dists[t].cdf(0.5)
                    cdf_0_5.append(cdf_t)
                cdf_0_5 = torch.stack(cdf_0_5, dim=1)
                p0 = cdf_0_5
            else:
                pdf0 = []
                for t in range(len(z_dists)):
                    pdf_t = z_dists[t].pdf(0.)
                    pdf0.append(pdf_t)
                pdf0 = torch.stack(pdf0, dim=1)  # [B, T, 1]
                p0 = pdf0

        p0 = p0.squeeze(-1)
        p0 = torch.where(mask, p0, p0.new_zeros([1]))  # [B, T]

        # L0 regularizer
        pdf_nonzero = 1. - p0  # [B, T]
        pdf_nonzero = torch.where(mask, pdf_nonzero, pdf_nonzero.new_zeros([1]))

        l0 = pdf_nonzero.sum(1) / (lengths + 1e-9)  # [B]
        l0 = l0.sum() / batch_size

        # `l0` now has the expected selection rate for this mini-batch
        # we now follow the steps Algorithm 1 (page 7) of this paper:
        # https://arxiv.org/abs/1810.00597
        # to enforce the constraint that we want l0 to be not higher
        # than `selection` (the target sparsity rate)

        # lagrange dissatisfaction, batch average of the constraint
        # c0_hat = l0 - selection
        if self.cfg['abs']:
            c0_hat = torch.abs(l0 - selection)
        else:
            c0_hat = l0 - selection

        # moving average of the constraint
        self.c0_ma = self.lagrange_alpha * self.c0_ma + \
            (1 - self.lagrange_alpha) * c0_hat.item()

        # compute smoothed constraint (equals moving average c0_ma)
        c0 = c0_hat + (self.c0_ma.detach() - c0_hat.detach())

        # update lambda
        self.lambda0 = max(self.lambda0 * torch.exp(self.lagrange_lr * c0.detach()), self.lambda_min)

        with torch.no_grad():
            optional["cost0_l0"] = l0.item()
            optional["target0"] = selection
            optional["c0_hat"] = c0_hat.item()
            optional["c0"] = c0.item()  # same as moving average
            optional["lambda0"] = self.lambda0.item()
            optional["lagrangian0"] = (self.lambda0 * c0_hat).item()
            optional["a"] = z_dists[0].a.mean().item()
            optional["b"] = z_dists[0].b.mean().item()

        loss = ce + self.lambda0.detach() * c0

        # z statistics
        if self.training:
            num_0, num_c, num_1, total = get_z_stats(self.latent_model.z, mask, strategy=self.strategy)
            optional["p0"] = num_0 / float(total)
            optional["pc"] = num_c / float(total)
            optional["p1"] = num_1 / float(total)
            optional["selected"] = 1 - optional["p0"]

        return loss, optional