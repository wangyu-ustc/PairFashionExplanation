import os
import torch
import numpy as np



class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer, negative_sampling_rate=1, data=None, debug=False, concat=False, return_idx=False):
        self.path = path
        self.negative_sampling_rate = negative_sampling_rate
        self.tz = tokenizer
        self.concat = concat
        if data is None:
            self.data = np.load(f"{self.path}data_filtered.npy")
        else:
            self.data = data
        self.return_idx = return_idx

        with open(f"{self.path}titles_exclude_nouns.txt", 'r', encoding='utf-8') as file:
            descriptions = file.read()
        file.close()

        with open(f"{self.path}categories_filtered.txt", "r", encoding='utf-8') as file:
            categories = file.read()
        file.close()

        self.descriptions = descriptions.strip().split("\n")
        
        if os.path.exists(f"{self.path}nouns.txt"):
            with open(f"{self.path}nouns.txt", 'r+') as file:
                self.nouns = file.read().strip().split("\n")
            file.close()
            self.nouns = [noun.split(" ") for noun in self.nouns]
        else:
            self.nouns = None
        
        self.categories = [category.split(", ") for category in categories.strip().split("\n")]
        
        # print("Building Tree...")
        # self.category_tree = build_tree_from_json(f"{self.path}tree.json")
        
        self.max_length = 200
        self.max_idx = len(self.descriptions)

        print(f"{len(self.data)} pairs and {len(self.descriptions)} items.")

        # Debug
        if debug:
            idxes = np.random.choice(np.arange(len(self.data)), replace=False, size=(500,))
            self.data = self.data[idxes]

        print("Initialization of Dataset Done")

    def __getitem__(self, i):

        if i < len(self.data):
            try:
                idx1, idx2 = self.data[i]
            except:
                idx1, idx2 = self.data(i)
            desc1, desc2, label = self.descriptions[idx1], self.descriptions[idx2], 1
        else:
            i = i % len(self.data)
            idx1, _ = self.data[i]
            selected_items = self.data[:, 1][np.where(self.data[:, 0] == idx1)]
            idx2 = np.random.choice(np.arange(len(self.descriptions)))
            # Exclude same category
            while idx2 in selected_items or self.categories[idx2][-1] == self.categories[idx1][-1]:
                idx2 = np.random.choice(np.arange(len(self.descriptions)))
            desc1, desc2, label = self.descriptions[idx1], self.descriptions[idx2], 0

        if self.nouns is not None:
            desc1 = desc1.split(" ")
            new_desc1 = []
            for word in desc1:
                if word in self.nouns[idx1]:
                    continue
                else:
                    new_desc1.append(word)
                
            desc1 = " ".join(new_desc1)
            
            desc2 = desc2.split(" ")
            new_desc2 = []
            for word in desc2:
                if word in self.nouns[idx2]:
                    continue
                else:
                    new_desc2.append(word)
            desc2 = " ".join(new_desc2)
        
        if not self.concat:
            desc1 = self.tz(desc1)['input_ids'][1:-1]
            desc2 = self.tz(desc2)['input_ids'][1:-1]
            if len(desc1) > self.max_length:
                desc1 = desc1[: self.max_length]
            else:
                desc1.extend([50257] * (self.max_length - len(desc1)))

            if len(desc2) > self.max_length:
                desc2 = desc2[: self.max_length]
            else:
                desc2.extend([50257] * (self.max_length - len(desc2)))
            if self.return_idx:
                return (torch.tensor(idx1), torch.tensor(idx2), torch.tensor(desc1), torch.tensor(desc2), label)
            else:
                return (torch.tensor(desc1), torch.tensor(desc2), label)
                
        else:
            desc1 = self.tz(desc1)['input_ids'][1:]
            desc2 = self.tz(desc2)['input_ids'][1:-1]

            if len(desc1) - 1 > self.max_length:
                desc1 = desc1[: self.max_length] + desc[-1:]
            if len(desc2) > self.max_length:
                desc2 = desc2[: self.max_length]

            desc = desc1 + desc2 

            desc.extend([50257] *((self.max_length * 2 + 1) - len(desc)))

            if self.return_idx:
                return (torch.tensor(idx1), torch.tensor(idx2), torch.tensor(desc), label)
            else:
                return (torch.tensor(desc), label)


    def __len__(self):
        return len(self.data) * (1 + self.negative_sampling_rate)





