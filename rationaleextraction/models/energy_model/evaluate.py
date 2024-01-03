import torch
from collections import defaultdict
from itertools import count
import numpy as np

from models.energy_model.util import get_minibatch, prepare_minibatch, prepare_minibatch_adv
from models.common.util import get_z_stats
from sklearn.metrics import f1_score


def get_histogram_counts(z=None, mask=None, mb=None):
    counts = np.zeros(5).astype(np.int64)

    for i, ex in enumerate(mb):

        tokens = ex.tokens
        token_labels = ex.token_labels

        if z is not None:
            ex_z = z[i][:len(tokens)]

        if mask is not None:
            assert mask[i].sum() == len(tokens), "mismatch mask/tokens"

        for j, tok, lab in zip(count(), tokens, token_labels):
            if z is not None:
                if ex_z[j] > 0:
                    counts[lab] += 1
            else:
                counts[lab] += 1

    return counts

def evaluate(model, loader, tokenizer, batch_size=25, device=None):
    """Accuracy of a model on given data set (using minibatches)"""

    model.eval()  # disable dropout

    # z statistics
    totals = defaultdict(float)
    z_totals = defaultdict(float)
    histogram_totals = np.zeros(5).astype(np.int64)
    z_histogram_totals = np.zeros(5).astype(np.int64)

    for idx, batch in enumerate(loader):
        
        x, targets = batch
        
        prepared_batch = prepare_minibatch(x, targets, device, return_reverse_map=True, concat=True)

        if prepared_batch is None: continue

        x, mask, targets, reverse_map = prepared_batch

        batch_size = targets.size(0)
        with torch.no_grad():

            logits,_,_ = model(x)  # forward pass
            loss, loss_optional = model.get_loss(logits, targets, mask=mask)
            predictions = model.predict(logits)
            
            if isinstance(loss, dict):
                loss = loss["main"]

            totals['loss'] += loss.item() * batch_size

            for k, v in loss_optional.items():
                if not isinstance(v, float):
                    v = v.item()

                totals[k] += v * batch_size

            if hasattr(model, "z"):
                n0, nc, n1, nt = get_z_stats(model.z, mask)
                z_totals['p0'] += n0
                z_totals['pc'] += nc
                z_totals['p1'] += n1
                z_totals['total'] += nt

                # histogram counts
                # for this need to sort z in original order
                z = model.z.squeeze(1).squeeze(-1)[reverse_map]
                mask = mask[reverse_map]

        # add the number of correct predictions to the total correct
        totals['acc'] += (predictions == targets.view(-1)).sum().item()
        totals['f1'] += f1_score(targets.view(-1).cpu(), predictions.view(-1).cpu(), average='micro')
        totals['total'] += batch_size

    result = {}

    # loss, accuracy, optional items
    totals['total'] += 1e-9
    for k, v in totals.items():
        if k != "total":
            result[k] = v / totals["total"]

    # z scores
    z_totals['total'] += 1e-9
    for k, v in z_totals.items():
        if k != "total":
            result[k] = v / z_totals["total"]

    if "p0" in result:
        result["selected"] = 1 - result["p0"]

    return result

def evaluate(model, data, batch_size=25, device=None):
    """Accuracy of a model on given data set (using minibatches)"""

    model.eval()  # disable dropout

    # z statistics
    totals = defaultdict(float)
    z_totals = defaultdict(float)
    histogram_totals = np.zeros(5).astype(np.int64)
    z_histogram_totals = np.zeros(5).astype(np.int64)

    for mb in get_minibatch(data, batch_size=batch_size, shuffle=False):
        x, targets, bias_targets, reverse_map = prepare_minibatch_adv(mb, model.bias_latent_model.vocab, device=device)
        mask = (x != 1)
        batch_size = targets.size(0)
        with torch.no_grad():

            outcome_prediction_logits, z_outcome, z_outcome_logits, bias_prediction_logits, z_bias, z_bias_logits = model(x)  # forward pass

            logits=outcome_prediction_logits
            predictions = model.predict(logits)

            loss, loss_optional = model.get_loss(outcome_prediction_logits, 
                                                bias_prediction_logits, 
                                                targets, 
                                                BIAS_THRED_STRATEGY='hyperparameter',
                                                BIAS_THRED=0.5,
                                                mask=mask)
            bias_prediction_logits = model.get_bias_prediction(x)  # forward pass

            if isinstance(loss, dict):
                loss = loss["main"]

            totals['loss'] += loss.item() * batch_size

            for k, v in loss_optional.items():
                if not isinstance(v, float):
                    v = v.item()

                totals[k] += v * batch_size

            if hasattr(model, "z_bias"):
                n0, nc, n1, nt = get_z_stats(model.z_bias, mask)
                z_totals['z_bias p0'] += n0
                z_totals['z_bias pc'] += nc
                z_totals['z_bias p1'] += n1
                z_totals['total'] += nt

                # histogram counts
                # for this need to sort z in original order
                # z = model.z.squeeze(1).squeeze(-1)[reverse_map]
                # mask = mask[reverse_map]
                # z_histogram = get_histogram_counts(z=z, mask=mask, mb=mb)
                # z_histogram_totals += z_histogram
                # histogram = get_histogram_counts(mb=mb)
                # histogram_totals += histogram
            if hasattr(model, "z_outcome"):
                n0, nc, n1, nt = get_z_stats(model.z_outcome, mask)
                z_totals['z_outcome p0'] += n0
                z_totals['z_outcome pc'] += nc
                z_totals['z_outcome p1'] += n1

        # add the number of correct predictions to the total correct
        totals['outcome acc'] += (predictions == targets.view(-1)).sum().item()
        totals['bias acc'] += ( model.predict(bias_prediction_logits) == bias_targets.view(-1)).sum().item()
        totals['outcome f1'] += f1_score(targets.view(-1).cpu(), predictions.view(-1).cpu(), average='micro')
        totals['bias f1'] += f1_score(bias_targets.view(-1).cpu(), model.predict(bias_prediction_logits).view(-1).cpu(), average='micro')
        totals['total'] += batch_size

    result = {}

    # loss, accuracy, optional items
    totals['total'] += 1e-9
    for k, v in totals.items():
        if k != "total":
            result[k] = v / totals["total"]

    # z scores
    z_totals['total'] += 1e-9
    for k, v in z_totals.items():
        if k != "total":
            result[k] = v / z_totals["total"]

    if "z_outcome p0" in result:
        result["selected"] = 1 - result["z_outcome p0"]

    return result


def generate_rationale(model, data, cfg, name="train", batch_size=25, device=None):
    """Accuracy of a model on given data set (using minibatches)"""

    model.eval()  # disable dropout
    outcome_i2t = ["non-toxic", "toxic"]
    bias_i2t = ["male", "female","trans", "other"]
    
    # z statistics
    if not os.path.exists(cfg["save_path"]):
            os.makedirs(cfg["save_path"])
    save_file=os.path.join(cfg["save_path"], name+'.txt')
    print("Write results to ", save_file)
    f=open(save_file, "w", errors='surrogatepass')
    for mb in get_minibatch(data, batch_size=batch_size, shuffle=False):
        x, targets, adv_targets, reverse_map = prepare_minibatch_adv(mb, model.vocab, device=device)
        sort_idx=np.argsort(reverse_map)
        mask = (x != 1)
        batch_size = targets.size(0)
        with torch.no_grad():

            z, z_bias = model.generate_rationale(x)
            z_bias = torch.where(mask, z_bias, z_bias.new_full([1], 1e2))
            z = torch.where(mask, z, z.new_full([1], 1e2))
            for i in range(z_outcome.shape[0]):
                d=mb[sort_idx[i]]
                f.write("====Example: "+str(i) + ", Bias "+ bias_i2t[d.adv_label] + ", Label "+ outcome_i2t[d.label] +"\n")
                f.write("*Original: "+ " ".join(d.tokens)+"\n")
                bias_rationale=[d.tokens[j] if z_bias[i][j]==1.0 else "_" for j in range(len(z_bias[i]))]
                f.write("*Bias Rationale: " + " ".join(bias_rationale)+"\n")
                final_rationale=[d.tokens[j] if z[i][j]==1.0 else "_" for j in range(len(z[i]))]
                f.write("*Debiased Rationale: " + " ".join(final_rationale)+"\n")
    f.close()