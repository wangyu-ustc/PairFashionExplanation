import numpy as np
import pdb

paths = [
    '../data/CSJ_shrinked/Clothing_Shoes_and_Jewelry_',
    '../data/FashionVCdata/'
]


data_filtered = None
all_data_for_debug = []
for path in paths:
    data = np.load(f"{path}data_filtered.npy")
    all_data_for_debug.append(data.copy())
    if data_filtered is None:
        data_filtered = data
    else:
        data += np.max(data_filtered) + 1
        data_filtered = np.concatenate([data_filtered, data])

descriptions = []
categories = []


for path in paths:

    with open(f"{path}titles_exclude_nouns.txt", 'r') as file:
        descriptions.extend(file.read().strip().split("\n"))
    file.close()

    with open(f"{path}categories_filtered.txt", "r") as file:
        categories.extend(file.read().strip().split("\n"))
    file.close()

np.save("../data/All/data_filtered.npy", data_filtered)

with open("../data/All/titles_exclude_nouns.txt", 'w') as file:
    file.write("\n".join(descriptions))
file.close()

with open("../data/All/categories_filtered.txt", 'w') as file:
    file.write("\n".join(categories))
file.close()
