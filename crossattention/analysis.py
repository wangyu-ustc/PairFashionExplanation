
import json
from tqdm import tqdm
import numpy as np

# path = 'FashionVC_fix_embed'
# path = 'CSJ_fix_embed'
path = 'All_fix_emb_cont'
# path = 'All_random'
elle_scores = json.load(open(f"./logs/{path}/elle_scores_mean.json", 'r'))
# scores_top2 = json.load(open(f"./logs/{path}/scores_2.json", 'r'))
scores = json.load(open(f"./logs/{path}/scores_-1_mean.json", 'r'))


print("ELLE scores:", np.mean(elle_scores))
# print(f"{path} Top2:", np.mean(scores_top2))
print(f"{path} All:", np.mean(scores))


elle_scores = np.sort(elle_scores)
scores = np.sort(scores)

ranks = []
idx = 0

for score in tqdm(elle_scores):

    while idx < len(scores) and scores[idx] < score:
        idx += 1
    
    ranks.append(idx)

print("ELLE mean ranks:", np.mean(ranks))
print("length of scores:", len(scores))








