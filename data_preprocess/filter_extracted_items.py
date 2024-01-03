import pdb
import pandas as pd

# filter the existing corpus
    
# load all the sentences
statistics = pd.read_csv("./data/data2.csv")
sentences = set([sen.strip() for sen in statistics['sentences'].values])

file = open("./results/responses_extracting_items.txt", 'r', encoding='utf-8')
current_results = file.read()
file.close()

paragraphs = []
contexts = []
unique_contexts = set()
for paragraph in current_results.strip().split("\n\n"):
    paragraph = paragraph.strip()
    if len(paragraph) == 0: continue
    if "Key features:" not in paragraph: continue
    if "Description:" not in paragraph: continue

    start = 0
    while not paragraph.split("\n")[start].startswith("Description:"):
        start += 1

    line = paragraph.strip().split("\n")[start]
    context = "Description: " + line.replace("Description:", "").strip()

    # If the sentence is not in the pool, then skip the current one
    if line.replace("Description:", "").strip() not in sentences:
        continue

    # If the sentence appeared twice, then skip this one
    if context not in unique_contexts:
        unique_contexts.add(context)
    else:
        continue

    paragraph = paragraph.strip().split("\n")[start+1:]
    if len(paragraph) == 0:
        continue
    
    new_paragraph = []
    for line in paragraph:
        if "Description:" in line:
            break
        else:
            new_paragraph.append(line)
    new_paragraph = '\n'.join(new_paragraph)
    paragraphs.append(context + '\n' + new_paragraph)
    contexts.append(context)

print("All contexts:", len(contexts))

with open("./results/responses_extracting_items_new.txt", 'w') as file:
    for paragraph in paragraphs:
        try:
            for line in paragraph.split("\n"):
                try:
                    file.write(line + '\n')
                except:
                    pass

            file.write("\n")
        except:
            pdb.set_trace()
file.close()
