from regex import F
import torch
from collections import defaultdict
from itertools import count
import numpy as np

from latent_rationale.sst.util import get_minibatch, prepare_minibatch
from latent_rationale.common.util import get_z_stats


logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')
# let's make this more or less deterministic (not resistent to restarts)
random.seed(12345)
np.random.seed(67890)
torch.manual_seed(10111213)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def evaluate(classifier, data, batch_size=25, device=None):
    """Accuracy of a model on given data set (using minibatches)"""

    classifier.eval()  # disable dropout

    # z statistics
    totals = defaultdict(float)
    z_totals = defaultdict(float)

    for mb in get_minibatch(data, batch_size=batch_size, shuffle=False):
        x, targets, reverse_map = prepare_minibatch(mb, model.vocab, device=device)
        mask = (x != 1)
        batch_size = targets.size(0)
        with torch.no_grad():

            # prediction on entire sentences
            logits = model(x)
            predictions = model.predict(logits)

            # prediction on rationale only


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
                z_histogram = get_histogram_counts(z=z, mask=mask, mb=mb)
                z_histogram_totals += z_histogram
                histogram = get_histogram_counts(mb=mb)
                histogram_totals += histogram

        # add the number of correct predictions to the total correct
        totals['acc'] += (predictions == targets.view(-1)).sum().item()
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

def faithfulness(classifier, results, batch_size=100, device="cpu"):
    def get_scores(input, labels):
        classifier.eval()
        with torch.no_grad():
            pred_list, probability_list  = [], []
            for batch_start in range(0, len(input), batch_size):
                d = results[batch_start:batch_start + batch_size]
                pred, logits = classifier(input)
                probability=F.softmax(logits, dim=1)
                pred_list.extend(pred)
                probability_list.extend(probability)
        return pred_list, probability_list

    original_sens=[results[i]['sen'] for i in range(len(results))]
    labels=[results[i]['label'] for i in range(len(results))]
    evidence_only_docs=[results[i]['rationale_mask'] for i in range(len(results))]
    rest_docs=[results[i]['rest_mask'] for i in range(len(results))]
    
    original_pred_list, original_probability_list=get_scores(original_sens, labels)
    evidence_pred_list, evidence_probability_list=get_scores(evidence_only_docs, labels)
    rest_pred_list, rest_probability_list=get_scores(rest_docs, labels)
    
    for i, (pred, prob) in enumerate(zip(original_pred_list, original_probability_list)) :
        classification_scores = {class_deinterner[cls]: p for cls, p in enumerate(pred)}
        rats[i]['sufficiency_classification_scores'] = classification_scores
    
    for i, ((ann_id, _), pred, _) in enumerate(zip(ids, zip(queries, evidence_only_docs))):
        classification_scores = {class_deinterner[cls]: p for cls, p in enumerate(pred)}
        rats[i]['sufficiency_classification_scores'] = classification_scores

