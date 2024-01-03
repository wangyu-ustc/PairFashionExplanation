import pdb
import json
import pandas as pd




responses = json.load(open("./responses_step1.json", 'r'))

print("length:", len(responses))

all_descriptions = []
all_items = []
all_features = []

for response in responses:
    items = []
    features = []

    if '$' in response['response']:
        continue

    for line in response['response'].split("Description")[0].split("\n"):
        line = line.strip()
        if ": " not in line:
            continue
        
        if "not specified" in line.lower() or "unknown" in line.lower():
            continue

        if not line.startswith("- "):
            continue

        item = line.split(": ")[0][2:]

        feature = line.split(": ")[1].replace(";", "")
        if feature.endswith("."):
            feature = feature[:-1]
        
        new_feature = []
        for word in feature.split(" "):
            if word == item:
                continue
            new_feature.append(word)
        feature = " ".join(new_feature)

        if '"' in feature:
            feature = feature.replace('"', "")

        if len(feature) == 0:
            continue

        items.append(item)
        features.append(feature)
    
    if len(items) < 2:
        continue

    description = response['prompt'].split("\n")[-3].split("Description: ")[1]
    if "$" in description: continue
    if ".com" in description: continue

    # Remove all the sentences talking about prices
    price_flag = False
    for item in items:
        if "price" in item.lower():
            price_flag = True
            break
    if price_flag:
        continue


    items = [item.lower() for item in items]
    all_items.append(items)
    all_features.append(features)
    all_descriptions.append(description.replace("  ", " "))

data = pd.DataFrame({
    'sentences': all_descriptions,
    'items': all_items,
    'features': all_features
})

print("Left data:", len(data))
# print(data)

data.to_csv("../data/data2.csv", index=False)


