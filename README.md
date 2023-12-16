This is the official implementation of our AAAI paper **Deciphering Compatibility Relationships with Textual Descriptions via Extraction
and Explanation**. 

<!-- make contents below -->


## Requirements
+ torch = 2.0.1
+ python 3.9.13
+ spacy 3.7.2

## Datasets
In our paper, we have two stages: Stage I: Extraction and Stage II: Explanation.
For the Stage I, we use the following dataset: 

**Amazon Reviews**: We downloaded the dataset from [here](https://nijianmo.github.io/amazon/index.html), where the category `Clothing Shoes and Jewelry` is chosen in our experiments. For each item, we leverage the `title` attribute and the `also_buy` list to build matching pairs. 

**FashionVC**: This dataset could be downloaded from [here](https://drive.google.com/file/d/1d72E3p4w280-vdCKfXtXZLMULpIScRRR/view). The FashionVC dataset includes 20,715 pairs of tops and bottoms, consisting of 14,871 tops and 13,663 bottoms. Here, each top or bottom possesses a corresponding category and feature, which could be used as $c_i$ and $t_i$ for training. 

For Stage II, we use our curated dataset `PFE-dataset` (Pair-Fashion-Explanation), which is shown in file `Pair-Fashion-Explanation.csv`. 

## Data Preprocessing of PFE-dataset
The detailed process steps are:  
### Dictionary of Categories:
The dictionary extracted from Amazon categories and re-
vised manually is shown in the following table: 
{'dress', 'top', 'skirt', 'jacket', 'shirt', 'pant', 'boot', 'jean',
'jeans', 'bag', 'coat', 'suit', 'trouser', 'blazer', 'trousers',
'sweater', 'shoe', 'blouse', 'shorts', 'shoes', 'sneaker', 'sandal',
'belt', 'pump', 'hat', 'scarf', 'leggings', 'necklace', 'bra',
'vest', 'cardigan', 'gown', 'loafer', 'sock', 'sunglass', 'handbag',
'sweatshirt', 'bodysuit', 'miniskirt', 'velvet', 'tote', 'satin',
'wool', 'jumpsuit', 'cloth', 'bracelet', 'plaid', 'cap', 'hoodie',
'corset', 'block', 'uniform', 'watch', 'cover', 'jumper', 'sundress',
'robe', 'clothes', 'pack', 'bustier', 'swimsuit', 'bootie', 'overalls',
'clog', 'shawl', 'slipper', 'lingerie', 'beret', 'fedora', 'pullover',
'costume', 'slingback', 'sweatpants', 'beanie', 'backpack', 'lounge',
'ballerina', 'espadrille', 'panty', 'windbreaker', 'kilt', 'waistcoat',
'leotard', 'saddle', 'brogue', 'pantyhose', 'jumpsuits', 'culotte',
'pouch', 'kimono', 'caftan', 'moccasin', 'bloomer', 't-shirt',
'briefcase', 'visor', 'sari', 'underwear', 'wallet', 'cloche',
'duffel', 'swimwear', 'panama', 'slip-on', 'ballgown', 'satchel'}

The percentages of these items are shown in the following table:  
{dress: 9.10%; top: 6.99%; skirt: 6.89%; jacket: 6.01%; shirt: 4.99%;
pant: 4.64%; boot: 4.12%; jean: 4.01%; jeans: 3.79%; bag: 3.11%; coat:
3.07%; suit: 2.94%; trouser: 2.77%; blazer: 2.75%; trousers: 2.50%;
sweater: 2.45%; shoe: 2.20%; blouse: 1.97%; shorts: 1.86%; shoes:
1.77%; sneaker: 1.70%; sandal: 1.58%; belt: 1.50%; pump: 1.39%; hat:
0.98%; scarf: 0.91%; leggings: 0.84%; necklace: 0.81%; bra: 0.79%;
vest: 0.77%; cardigan: 0.77%; gown: 0.67%; loafer: 0.65%; sock: 0.63%;
sunglass: 0.60%; handbag: 0.45%; sweatshirt: 0.41%; bodysuit: 0.40%;
miniskirt: 0.38%; velvet: 0.34%; tote: 0.27%; satin: 0.27%; wool:
0.27%; jumpsuit: 0.26%; cloth: 0.25%; bracelet: 0.23%; plaid: 0.22%;
cap: 0.21%; hoodie: 0.21%; corset: 0.21%; block: 0.17%; uniform:
0.17%; watch: 0.17%; cover: 0.14%; jumper: 0.14%; sundress: 0.13%;
robe: 0.13%; clothes: 0.13%; pack: 0.11%; bustier: 0.11%; swimsuit:
0.11%; bootie: 0.10%; overalls: 0.09%; clog: 0.09%; shawl: 0.08%;
slipper: 0.08%; lingerie: 0.08%; beret: 0.08%; fedora: 0.08%; pullover:
0.07%; costume: 0.06%; slingback: 0.06%; sweatpants: 0.05%; beanie:
0.05%; backpack: 0.05%; lounge: 0.04%; ballerina: 0.04%; espadrille:
0.04%; panty: 0.04%; windbreaker: 0.04%; kilt: 0.03%; waistcoat:
0.03%; leotard: 0.03%; saddle: 0.02%; brogue: 0.02%; pantyhose: 0.02%;
jumpsuits: 0.02%; culotte: 0.02%; pouch: 0.02%; kimono: 0.02%; caftan:
0.01%; moccasin: 0.01%; bloomer: 0.01%; t-shirt: 0.01%; briefcase:
0.01%; visor: 0.01%; sari: 0.01%; underwear: 0.01%; wallet: 0.01%;
cloche: 0.01%; duffel: 0.01%; swimwear: 0.01%; panama: 0.01%; slip-on:
0.01%; ballgown: 0.01%; satchel: 0.01% }

### Prompts to query ChatGPT
To filter out irrelevant sentences, we adopt the package `spacy` to
perform named entity recognition (NER) on the sentence,
extracting the entities with the NER tag "Noun". We obtain 48,726
sentences. Then we turn to GPT-3.5-turbo for filtering. The
prompt for the first cycle of filtering is as follows:

```
Let me give you an example:
Description: The hits run from Poirets stunning 1919 opera coat, made of a single swath of uncut purple silk velvet, to a 2018 kimono printed with
oversize manga characters by Comme des Garons founder Rei Kawakubo.
Items: coat, kimono;
Key features:
- coat: purple silk velvet;
- kimono: oversize manga characters;
Then please give the key features concisely following the above structure
in the following question:
Description: {sentence}
Items: {items}
Key features:
```
where the sentence and items are the sentence from the dataset and the entities extracted with NER.  
After querying ChatGPT, we end up with all the features
corresponding to the extracted items, where some of key features are denoted as “not-specified”, meaning that ChatGPT could not find the key features. Such sentences are dropped.
Then we have 37737 sentences left. Then we query ChatGPT again with the following
prompt: 
Then we ask gpt-3.5-turbo with the following prompt:  
```
Could you read this sentence and let me know if it is explaining why two pieces of clothing look good together as an outfit? It's possible that the sentence could be one or several sentences about why the two items complement each other and create a cohesive outfit. If no, then simply answer a "No"; If yes, please give a concise reason for how they complement each other in the form of "Reason: They match because ...".
{sentence}
```
where the sentence is the extracted sentence from the above 37737 sentences. With the returned answers, we filter out the sentence with the answer ”No”. For the left sentences, we construct a new dataset with the entities, and features extracted from the previous and the sentences rewritten by ChatGPT with the answer "Yes". After the above process, we obtain a dataset with 6,407 examples.

### Implementation and Running Scripts

We first crawl data from public magazines searched with the key phrases: {"Clothes Match", "Clothes Fashion", "Fashion", "Outfit of The Day", "Style", "Match Clothes", "How to Dress", "How to Wear", "Match"}. 
We are not releasing the data crawled from the magazines and Youtube as these files are huge and dirty. 
Then with the file `data_preprocess/process_step1.py`, we could obtain the file `data/data1.csv`. 

After this, we ask `gpt-3.5-turbo` to extract the related features according to the sentence and the corresponding nouns (items). The responses quried from ChatGPT is shown in `data_preprocess/responses.json`. With this file, we could run `data_preprocess/process_step2.py` to obtain the file `data/data2.csv`. 

Then we query ChatGPT with  the obtained results are saved in `./data_preprocess/responses_step2.json`. Then we extract the answers out and discard all the samples with the response `No`, yielding the dataset `data.csv`. 

## Implementation Details
The configurations for fine-tuning adapted languages models are: learning rate=0.0002, weight decay=0, optimizer=Adam, training epoch=20, batch size=5 for GPT2
and Flan-T5-large, while batch size=1 for Flan-t5-xl, max length=100, All codes are implemented with Python3.9.12 and PyTorch2.0.1 with CUDA 11.7. operated on Ubuntu (16.04.7 LTS) server with 2 NVIDIA GeForce GTS A6000 GPUs. Each has a memory of 49GB.

### CrossAttention Extractor
### Cross Attention Extractor
With $t_i$ and $t_j$, we use `Bert-tokenizer` to tokenize the sentences and get the embedding for each word $\{\textbf{e}_{i1},\cdots,\textbf{e}_{il_i}\}$ and $\{\textbf{e}_{j1},\cdots,\textbf{e}_{jl_j}\}$, where $l_i$ and $l_j$ are the lengths of the indices of tokenized sentences $t_i$ and $t_j$. $\{\textbf{e}_{i1},\cdots, \textbf{e}_{il_i}\}$, $\{\textbf{e}_{j1},\cdots,\textbf{e}_{jl_j}\}$ are the embeddings for $t_i$ and $t_j$, respectively, with each $\textbf{e} \in \mathbb{R}^{512}$.  
Then we calculate the attention map $\mathbf{A}\in\mathbb{R}^{l_i\times l_j}$ with each element as:

$$ a_{k_ik_j} = \textrm{Sigmoid}(\textrm{MLP}(\textrm{Cat}(\textbf{e}_{ik_i}, \textbf{e}_{jk_j}))),$$

$$k_i\in\{1,\cdots,l_i\}, k_j \in \{1,\cdots,l_j\}$$

Then we normalize the attention score so that all scores add up to 1:

$$\overline{\mathbf{A}} = \mathbf{A} / \textrm{Sum}(\mathbf{A})$$

where $\textrm{Sum}(\mathbf{A})$ means the sum of all the elements in $\mathbf{A}$. Then we could get the weighted average of the concatenated embeddings:

$$\textbf{e}_{\mathit{avg}} = \frac{1}{l_i * l_j} \sum_{k_i=1}^{l_i} \sum_{k_j=1}^{l_j} \overline{a}_{k_ik_j} \textrm{Cat}(\textbf{e}_{k_i}, \textbf{e}_{k_j})$$

where $\overline{a}_{k_ik_j}$ is the corresponding element in $\overline{\mathbf{A}}$ $\textbf{e}_{\mathit{avg}} \in \mathbb{R}^{1024}$. Then we stack another MLP $h_\phi$ on top of $\textbf{e}_{\mathit{avg}}$ to yield the prediction:

$$\hat{y} = h_\phi(\textbf{e}_{\mathit{avg}})$$

Subsequently, given the label of each pair $(t_i, t_j)$ as positive or negative, we utilize CrossEntropy loss to perform backward propagation and update the two MLPs in Eq. (1) and Eq. (2). Following training, we use the first MLP in Eq. (1) to derive the attention score $\mathbf{A}$ prior to normalization. Then we calculate the attention score of every word pair between $t_i$ and $t_j$. By averaging the attention scores across rows and columns, we identify the most significant words, $w_i$ in $t_i$ and $w_j$ in $t_j$, which inform the definition of $f_\theta$ in Cross-Attention:

$$w_i, w_j = f_\theta(t_i, t_j)$$


## Additioanl Experiments
### Case Study
We provide more case studies in the following tables. 

#### **Generated results of combining two stages together**
| **Item/Feature** | **skirt dress/lemonbaby Peppa pig cartoon girls cotton birthday baby; costume/Peppa pig t ballerina** |
| --- | --- |
| **PEPLER-F** | The peppa-colored skirt complements the pink and blue colors in the dress. the peppa-colored skirt also adds a pop of color to the outfit. |
| **ChatGPT** | The Lemonbaby Peppa Pig cartoon girls cotton birthday baby skirt dress and Peppa Pig T ballerina costume match because they both feature Peppa Pig as the main character and use similar color schemes and design elements. |
| **Flan-T5-xl-F** | The lemonbaby pig girls baby skirt dress and the peppa t ballerina costume complement each other because they both have a playful and playful vibe, which creates a cohesive outfit. |

| **Item/Feature** | **sock/cutieful women nylon mmhg compression dark flower power; shoe/cutieful therapeutic graduated mmhg compression dapper dogs women sizes socks** |
| --- | --- |
| **PEPLER-F** | they are both made of the same fabric and have a similar color scheme. they complement each other because they create a cohesive outfit by balancing the proportions of the outfit. the combination of a feminine and masculine look. |
| **ChatGPT** | Both Cutieful Women Nylon MMHG Compression Dark Flower Power Socks and Cutieful Therapeutic Graduated MMHG Compression Dapper Dogs Women Sizes Socks provide comfortable and stylish compression support for your legs and feet. |
| **Flan-T5-xl-F-RE** | They complement each other because the cuteness of the graduated dapper dogs socks adds a playful and playful element to the outfit, while the compression sock provides a more serious and structured look. |

| **Item/Feature** | **knot/sterling silver thick love post earrings; jewelry/lifetime round ball stud earrings inch mm wide k gold semi precious metals surgical steel posts safe for most ears guaranteed life** |
| --- | --- |
| **PEPLER-F** | the silver chain necklace and bracelets are both silver and have a polished finish that complements the silver chain necklace and bracelets. the silver chain necklace and bracelets are a great accessory for a statement piece. |
| **ChatGPT** | The reason why these earrings are great is that they are made from high-quality materials such as sterling silver, semi-precious metals, and surgical steel with a guaranteed lifetime. They are also designed to match any outfit making them a great addition to any jewelry collection. |
| **Flan-T5-xl-F-RE** | They complement each other because the sterling knot adds a touch of elegance and sophistication to the jewelry, while the lifetime round inch wide metal creates a sleek and polished look. |

#### **Generated results of Stage II**
| **Items** | **'jackets', 'wool capes'** |
| **Features** | **'tailored with roundness', 'made for interesting silhouettes and brought freshness to saturated tailoring landscape'** |
| --- | --- |
| **GPT2-F** | The pairing of a tailored jacket and layered tailoring creates a unique and interesting silhouette that is complemented by the addition of freshness to the wool capes. |
| **Flan-T5-large-F** | The roundness of the jackets complements the interesting silhouettes and wool capes add freshness to the saturated tailoring landscape. |
| **Flan-T5-xl-F** | They complement each other because the roundness of the jackets creates an interesting silhouette, while the wool capes add a freshness to the saturated tailoring landscape. |
| **ChatGPT** | Tailored round jackets and wool capes create unique and refreshing silhouettes, adding depth to the saturated tailoring trend. They match well together due to their similar structured styles. |
| **Ground Truth** | The roundness of the tailored jackets and wool capes complement each other and create interesting silhouettes, adding a fresh perspective to a saturated tailoring landscape. |

| **Items** | **'skirt', 'boots'** |
| **Features** | **'frilly blush', 'made of leather, feminine touch'** |
| --- | --- |
| **GPT2-F** | They complement each other because the feminine touch of the leather boots contrasts with the fun and playful vibe of the frilly skirt, creating an interesting and balanced outfit. |
| **Flan-T5-large-F** | The frilly blush skirt complements the feminine touch of the leather boots, creating a balanced outfit. |
| **Flan-T5-xl-F** | They complement each other because the frilly blush skirt adds a feminine touch to the masculine leather boots, creating a balanced outfit. |
| **ChatGPT** | The combination of a frilly blush skirt and leather boots with a feminine touch creates a balance of softness and edginess, making for a fashionable and unique outfit. |
| **Ground Truth** | They complement each other because the frilly blush skirt adds a hint of softness to the edginess of the leather boots, creating a balanced and harmonious outfit. |

| **Items** | **'shirt', 'pants'** |
| **Features** | **'ruffled button-down', 'black printed cigarette'** |
| --- | --- |
| **GPT2-F** | The ruffled button-down shirt adds a feminine touch to the outfit, while the black printed cigarette pants provide a neutral base that complements both items. |
| **Flan-T5-large-F** | The black printed cigarette pants and ruffled button-down shirt create a cohesive outfit that is both stylish and professional. |
| **Flan-T5-xl-F** | The ruffled button-down shirt and black printed cigarette pants complement each other because they create a contrast in textures and patterns, creating a visually interesting outfit. |
| **ChatGPT** | The combination of a ruffled button-down shirt and black printed cigarette pants match due to the contrast between the structured and feminine top and the sleek and edgy pants for a balanced yet bold outfit. |
| **Ground Truth** | The ruffled button-down shirt adds texture and volume to the outfit, while the black printed cigarette pants provide a sleek and sophisticated contrast. |





## Training

### Stage I: Extraction
We put the combined dataset in the folder `data/All`. The script for training `Cross-Attn` model is shown below: 

```
cd crossattention
python main.py --mode train --llm bert --fix_emb --save_path logs/
```
We could obtain a model with accuracy 0.8505 on validation set after one epoch. 

(... to be continued before the end of 2023)

