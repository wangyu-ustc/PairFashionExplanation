import pdb
import json
from tqdm import tqdm
import numpy as np

path = 'rationaleextraction_selection0.3'
elle_scores = json.load(open(f"./logs/{path}/elle_scores.json", 'r'))
elle_scores_max = json.load(open(f"./logs/{path}/elle_scores_max.json", 'r'))
# scores_top2 = json.load(open(f"./logs/{path}/scores_2.json", 'r'))
scores = json.load(open(f"./logs/{path}/scores_all.json", 'r'))






elle_scores = np.array(elle_scores)[np.where(1 - np.isnan(elle_scores))]
elle_scores_max = np.array(elle_scores_max)[np.where(1 - np.isnan(elle_scores_max))]
scores = np.array(scores)[np.where(1 - np.isnan(scores))]


print("ELLE scores:", np.mean(elle_scores))
print("ELLE max scores:", np.mean(elle_scores_max))
# print(f"{path} Top2:", np.mean(scores_top2))
print("{path} All:", np.mean(scores))


elle_scores = np.sort(elle_scores)
scores = np.sort(scores)

ranks = []
idx = 0

for score in tqdm(elle_scores):

    while scores[idx] < score:
        idx += 1
    
    ranks.append(idx)

print("ELLE mean ranks:", len(scores) - np.mean(ranks))
print("length of scores:", len(scores))


ranks = []
idx = 0
for score in tqdm(elle_scores_max):

    while scores[idx] < score:
        idx += 1
    
    ranks.append(idx)

print("ELLE Max mean ranks:", len(scores) - np.mean(ranks))
print("length of scores:", len(scores))