import os
import pdb
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import nltk.data
import json
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import *


def remove_non_ascii(string):
    return ''.join(char for char in string if ord(char) < 128)

def convert_html_to_txt():

    folders = ['Youtube'] # We are not releasing the data crawled from the magazines and Youtube as these files are too huge.

    for folder in folders:
        for file in os.listdir(f'./data/magazines/{folder}'):

            texts = []

            with open(f"./data/magazines/{folder}/{file}", 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            f.close()
            items = soup.find_all("div", 'css-53u6y8')
            for item in items:
                if item.find("p"):
                    texts.append(item.find('p').text)

            texts = ' '.join(texts)
            with open(f"./data/magazines/{folder}/{file.split('.')[0]}.txt", 'w', encoding='utf-8') as f:
                f.write(texts)
            f.close()


def filter_txt(magazines):
    
    intersection_nouns =  {'sweatpants', 'pack', 'backpack', 'flip-flop', 'rompers', 'scarf', 'cardigan', 'cloth',
    'panty', 'cloak', 'shorts', 'kilt', 'blouse', 'jumper', 'bracelet', 'jacket',
    'sweatshirt', 'sandal', 'sari', 'sombrero', 'velvet', 'windbreaker', 'bloomer', 'sleepwear', 'sneaker', 'lingerie',
    'shawl', 'hat', 'deerstalker', 'sweater', 'bag', 'culotte', 'saddle', 'pump', 'boot', 'bodysuit', 'daypack',
    'slipper', 'nightgown', 'beanie', 'lounge', 'shirt', 'kimono', 'pouch', 'waistcoat', 'skirt', 'visor', 'dress',
    'barrel', 'beret', 'watch', 'top', 'swimwear', 'cap', 'pompom', 'sunglass', 'trouser', 'sock', 'drawstring', 'leggings',
    'bustier', 'costume', 'caftan', 'satchel', 'belt', 'bra', 'jumpsuits', 'clog', 'ballgown', 'tote', 'overalls', 'corset',
    'wristlet', 'fedora', 'pullover', 'handbag', 'necklace', 'suit', 'hoisery', 'block', 'miniskirt', 'rucksack', 'espadrille',
    'moccasin', 'ballerina', 'brogue', 'bootie', 'cover', 'duffel', 'cone', 'derbi', 'pant', 't-shirt', 'parker', 'activewear',
    'hoodie', 'petit', 'sundress', 'briefcase', 'robe', 'shoe', 'leotard', 'trousers', 'clothes', 'vest', 'loafer', 'jeans',
    'dungaree', 'taper', 'plaid', 'jumpsuit', 'gumboot', 'wallet', 'wool', 'beachwear', 'bonnet', 'blazer', 'underwear',
    'swimsuit', 'slip-on', 'coat', 'cloche', 'slingback', 'uniform', 'shoes', 'knickerbocker', 'gown', 'panama', 'jean',
    'maternity', 'skate', 'pantyhose', 'satin'}

    # initialize tokenizer
    extra_abbreviations = ['\n']
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    tokenizer._params.abbrev_types.update(extra_abbreviations)
    word_tokenizer = RegexpTokenizer(r'\w+')
    stemmer = PorterStemmer()

    # initialize spacy
    import spacy
    from spacy.lang.en import English

    # need to run "python -m spacy download en_core_web_sm" first
    nlp = spacy.load("en_core_web_sm")

    aligned_sentences = []
    aligned_nouns = []
    
    for magazine in magazines:
        count = 0
        selected_count = 0
        all_sentences = set()
        folders = os.listdir(f"./data/magazines/{magazine}")

        with open(f"./data/magazines/{magazine}/filtered_corpus.txt", "w", encoding='utf-8') as f2:
            for folder in folders:
                if folder.endswith("txt"): continue
                for file in os.listdir(f'./data/magazines/{magazine}/{folder}'):
                    if file.endswith("txt"):
                        with open(f"./data/magazines/{magazine}/{folder}/{file}", 'r', encoding='unicode_escape') as f:
                            text = f.read()
                        f.close()

                        sentences = tokenizer.tokenize(text)
                        sentences = [sentence.strip() for sentence in sentences]

                        new_sentences = []
                        for sentence in sentences:
                            if '\n' in sentence:
                                new_sentences.extend(sentence.split("\n"))
                            else:
                                new_sentences.append(sentence)
                        sentences = new_sentences
                        
                        for sentence in sentences:
                            
                            sentence = remove_non_ascii(sentence)

                            if sentence in all_sentences:
                                continue

                            else:
                                all_sentences.add(sentence)
                            
                            # use spacy to get all the nouns
                            nouns = []
                            for x in nlp(sentence):
                                if x.pos_ == 'NOUN':
                                    word = x.text
                                    if word.lower() in intersection_nouns or stemmer.stem(word.lower()) in intersection_nouns:
                                        nouns.append(word.lower())
                            
                            nouns = list(set(nouns))
                            
                            if len(nouns) >= 2:
                                aligned_sentences.append(sentence)
                                aligned_nouns.append(nouns)
                                selected_count += 1
                            count += 1
                           

        print("For Magazine: ", magazine)
        print(f"Selected {selected_count} sentences out of {count} sentences.")
    return pd.DataFrame({
        "sentences": aligned_sentences,
        "nouns": aligned_nouns,
    })





if __name__ == '__main__':

    magazines = ['Youtube']

    # count_words(magazines)
    statistics = filter_txt(magazines)
    statistics.to_csv("./data/data1.csv", index=False)

