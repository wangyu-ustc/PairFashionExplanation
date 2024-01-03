#!/usr/bin/env python

import torch
from torch import nn
from models.common.util import get_z_stats
from models.common.classifier import Classifier
from models.common.latent import EPS, DependentLatentModel
from torch.nn.functional import softplus, sigmoid, tanh

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Model device:", device)


__all__ = ["LatentRationaleModel"]


class LatentRationaleModel(nn.Module):
    """
    Latent Rationale

    Consists of:

    p(y | x, z)     observation model / classifier
    p(z | x)        latent model

    In case of full VI:
    q(z | x, y)     inference model

    """
    def __init__(self,
                 vocab:          object = None,
                 vocab_size:     int = 0,
                 emb_size:       int = 200,
                 hidden_size:    int = 200,
                 output_size:    int = 1,
                 dropout:        float = 0.1,
                 layer:          str = "rcnn",
                 dependent_z:    bool = True,
                 z_rnn_size:     int = 30,
                 selection:      float = 1.,
                 lasso:          float = 0.,
                 lagrange_alpha: float = 0.5,
                 lagrange_lr:    float = 0.05,
                 lambda_init:    float = 0.0015,
                 lambda_min:     float = 1e-12,
                 lambda_max:     float = 5.,
                 ):

        super(LatentRationaleModel, self).__init__()

        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab = vocab

        self.selection = selection
        self.lasso = lasso

        self.lagrange_alpha = lagrange_alpha
        self.lagrange_lr = lagrange_lr
        self.lambda_init = lambda_init
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        self.z_rnn_size = z_rnn_size
        self.dependent_z = dependent_z

        self.embed = embed = nn.Embedding(vocab_size, emb_size, padding_idx=1)

        self.classifier = Classifier(
            embed=embed, hidden_size=hidden_size, output_size=output_size,
            dropout=dropout, layer=layer)

        if self.dependent_z:
            self.latent_model = DependentLatentModel(
                embed=embed, hidden_size=hidden_size,
                dropout=dropout, layer=layer)
        else:
            raise ValueError("Independent not implemented")

        self.criterion = nn.NLLLoss(reduction='none') # classification

        # lagrange buffers
        self.register_buffer('lambda0', torch.full((1,), lambda_init))
        self.register_buffer('lambda1', torch.full((1,), lambda_init))
        self.register_buffer('c0_ma', torch.full((1,), 0.))  # moving average
        self.register_buffer('c1_ma', torch.full((1,), 0.))  # moving average

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
    def forward(self, x):
        """
        Generate a sequence of zs with the Generator.
        Then predict with sentence x (zeroed out with z) using Encoder.

        :param x: [B, T] (that is, batch-major is assumed)
        :return:
        """
        mask = (x != 1)  # [B,T]
        z = self.latent_model(x, mask)
        y = self.classifier(x, mask, z)

        return y, z, z #! need to change it to y, z, logits

    def get_loss(self, preds, targets, mask=None):

        optional = {}
        # loss_mat = self.criterion(preds, targets)  # [B, T]

        # main MSE loss for p(y | x,z)
        # loss_vec = loss_mat.mean(1)   # [B]
        # mse = loss_vec.mean()         # [1]
        # optional["mse"] = mse.item()  # [1]

        loss_vec = self.criterion(preds, targets)  # [B]

        # main MSE loss for p(y | x,z)
        ce = loss_vec.mean()        # [1]
        optional["ce"] = ce.item() 

        batch_size = mask.size(0)
        lengths = mask.sum(1).float()  # [B]

        # z = self.generator.z.squeeze()
        z_dists = self.latent_model.z_dists

        # pre-compute for regularizers: pdf(0.)
        if len(z_dists) == 1:
            # pdf0 = z_dists[0].pdf(0.)
            cdf0 = z_dists[0].cdf(0.)
        else:
            # pdf0 = []
            # for t in range(len(z_dists)):
            #     pdf_t = z_dists[t].pdf(0.)
            #     pdf0.append(pdf_t)
            # pdf0 = torch.stack(pdf0, dim=1)  # [B, T, 1]
            cdf0 = []
            for t in range(len(z_dists)):
                cdf_t = z_dists[t].cdf(0.)
                cdf0.append(cdf_t)
            cdf0 = torch.stack(cdf0, dim=1)

        # pdf0 = pdf0.squeeze(-1)
        # pdf0 = torch.where(mask, pdf0, pdf0.new_zeros([1]))  # [B, T]
        
        cdf0 = cdf0.squeeze(-1)
        cdf0 = torch.where(mask, cdf0, cdf0.new_zeros([1]))  # [B, T]


        #===============================
        if len(z_dists) == 1:
            # pdf0 = z_dists[0].pdf(0.)
            cdf0_5 = z_dists[0].cdf(0.5)
        else:
            # pdf0 = []
            # for t in range(len(z_dists)):
            #     pdf_t = z_dists[t].pdf(0.)
            #     pdf0.append(pdf_t)
            # pdf0 = torch.stack(pdf0, dim=1)  # [B, T, 1]
            cdf0_5 = []
            for t in range(len(z_dists)):
                cdf_t = z_dists[t].cdf(0.5)
                cdf0_5.append(cdf_t)
            cdf0_5 = torch.stack(cdf0_5, dim=1)

        # pdf0 = pdf0.squeeze(-1)
        # pdf0 = torch.where(mask, pdf0, pdf0.new_zeros([1]))  # [B, T]
        
        cdf0_5 = cdf0_5.squeeze(-1)
        cdf0_5 = torch.where(mask, cdf0_5, cdf0_5.new_zeros([1]))  # [B, T]

    #===============================
        if len(z_dists) == 1:
            # pdf0 = z_dists[0].pdf(0.)
            cdf1 = z_dists[0].cdf(1-EPS)
        else:
            # pdf0 = []
            # for t in range(len(z_dists)):
            #     pdf_t = z_dists[t].pdf(0.)
            #     pdf0.append(pdf_t)
            # pdf0 = torch.stack(pdf0, dim=1)  # [B, T, 1]
            cdf1 = []
            for t in range(len(z_dists)):
                cdf_t = z_dists[t].cdf(1.0-EPS)
                cdf1.append(cdf_t)
            cdf1 = torch.stack(cdf1, dim=1)

        # pdf0 = pdf0.squeeze(-1)
        # pdf0 = torch.where(mask, pdf0, pdf0.new_zeros([1]))  # [B, T]
        
        cdf1 = cdf1.squeeze(-1)
        cdf1 = torch.where(mask, cdf1, cdf1.new_zeros([1]))  # [B, T]


        # L0 regularizer
        # pdf_nonzero = 1. - pdf0  # [B, T]
        # pdf_nonzero = torch.where(mask, pdf_nonzero, pdf_nonzero.new_zeros([1]))

        # l0 = pdf_nonzero.sum(1) / (lengths + 1e-9)  # [B]
        # l0 = l0.sum() / batch_size

        cdf_nonzero = 1. - cdf0  # [B, T]
        cdf_nonzero = torch.where(mask, cdf_nonzero, cdf_nonzero.new_zeros([1]))

        l0 = cdf_nonzero.sum(1) / (lengths + 1e-9)  # [B]
        l0 = l0.sum() / batch_size


        # `l0` now has the expected selection rate for this mini-batch
        # we now follow the steps Algorithm 1 (page 7) of this paper:
        # https://arxiv.org/abs/1810.00597
        # to enforce the constraint that we want l0 to be not higher
        # than `selection` (the target sparsity rate)

        # lagrange dissatisfaction, batch average of the constraint
        c0_hat = (l0 - self.selection)

        # moving average of the constraint
        self.c0_ma = self.lagrange_alpha * self.c0_ma + \
            (1 - self.lagrange_alpha) * c0_hat.item()

        # compute smoothed constraint (equals moving average c0_ma)
        c0 = c0_hat + (self.c0_ma.detach() - c0_hat.detach())

        # update lambda
        self.lambda0 = self.lambda0 * torch.exp(self.lagrange_lr * c0.detach())

        with torch.no_grad():
            optional["cost0_l0"] = l0.item()
            optional["target0"] = self.selection
            optional["c0_hat"] = c0_hat.item()
            optional["c0"] = c0.item()  # same as moving average
            optional["lambda0"] = self.lambda0.item()
            optional["lagrangian0"] = (self.lambda0 * c0_hat).item()
            optional["a"] = z_dists[0].a.mean().item()
            optional["b"] = z_dists[0].b.mean().item()

            optional["mean_cdf_0"]=cdf0.mean().item()
            optional["mean_cdf_0_5"]=cdf0_5.mean().item()
            optional["mean_cdf_1"]=cdf1.mean().item()

        loss = ce + self.lambda0.detach() * c0

        if self.lasso > 0.:
            # fused lasso (coherence constraint)

            # cost z_t = 0, z_{t+1} = non-zero
            # zt_zero = pdf0[:, :-1]
            # ztp1_nonzero = pdf_nonzero[:, 1:]
            zt_zero = cdf0[:, :-1]
            ztp1_nonzero = cdf_nonzero[:, 1:]


            # cost z_t = non-zero, z_{t+1} = zero
            # zt_nonzero = pdf_nonzero[:, :-1]
            # ztp1_zero = pdf0[:, 1:]

            zt_nonzero = cdf_nonzero[:, :-1]
            ztp1_zero = cdf0[:, 1:]

            # number of transitions per sentence normalized by length
            lasso_cost = zt_zero * ztp1_nonzero + zt_nonzero * ztp1_zero
            lasso_cost = lasso_cost * mask.float()[:, :-1]
            lasso_cost = lasso_cost.sum(1) / (lengths + 1e-9)  # [B]
            lasso_cost = lasso_cost.sum() / batch_size

            # lagrange coherence dissatisfaction (batch average)
            target1 = self.lasso

            # lagrange dissatisfaction, batch average of the constraint
            c1_hat = (lasso_cost - target1)

            # update moving average
            self.c1_ma = self.lagrange_alpha * self.c1_ma + \
                (1 - self.lagrange_alpha) * c1_hat.detach()

            # compute smoothed constraint
            c1 = c1_hat + (self.c1_ma.detach() - c1_hat.detach())

            # update lambda
            self.lambda1 = self.lambda1 * torch.exp(
                self.lagrange_lr * c1.detach())

            with torch.no_grad():
                optional["cost1_lasso"] = lasso_cost.item()
                optional["target1"] = self.lasso
                optional["c1_hat"] = c1_hat.item()
                optional["c1"] = c1.item()  # same as moving average
                optional["lambda1"] = self.lambda1.item()
                optional["lagrangian1"] = (self.lambda1 * c1_hat).item()

                

            loss = loss + self.lambda1.detach() * c1

        # z statistics
        if self.training:
            num_0, num_c, num_1, total = get_z_stats(self.latent_model.z, mask)
            optional["p0"] = num_0 / float(total)
            optional["pc"] = num_c / float(total)
            optional["p1"] = num_1 / float(total)
            optional["selected"] = 1 - optional["p0"]

        return loss, optional