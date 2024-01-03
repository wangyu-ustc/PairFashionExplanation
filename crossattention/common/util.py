import torch

def make_kv_string(d):
    out = []
    for k, v in d.items():
        if isinstance(v, float):
            out.append("{} {:.4f}".format(k, v))
        else:
            out.append("{} {}".format(k, v))

    return " ".join(out)


def build_model(cfg):

    lasso = cfg["lasso"]
    llm = cfg['llm']
    from models.latent_extractor import CrossAttentionModel
    model = CrossAttentionModel(
        cfg=cfg, output_size=2, lasso=lasso, llm=llm)
    return model

