#!/usr/bin/env python

from models.energy_model.models.latent import LatentRationaleModel

def build_model(cfg):

    emb_size = cfg["embed_size"]
    hidden_size = cfg["hidden_size"]
    layer = cfg["layer"]
    dependent_z = cfg.get("dependent_z", False)

    selection = cfg["selection"]

    assert 0 < selection <= 1.0, "selection must be in (0, 1]"

    selection = cfg["selection"]
    lagrange_alpha = cfg["lagrange_alpha"]
    lagrange_lr = cfg["lagrange_lr"]
    lambda_init = cfg["lambda_init"]
    strategy = cfg['strategy']
    return LatentRationaleModel(
        cfg=cfg, emb_size=emb_size, hidden_size=hidden_size,
        output_size=2,
        dependent_z=dependent_z, layer=layer,
        selection=selection,
        lagrange_alpha=lagrange_alpha, lagrange_lr=lagrange_lr,
        lambda_init=lambda_init, strategy=strategy)
