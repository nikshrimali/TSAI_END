import spacy
spacy_en = spacy.load('en')

import numpy as np
import pandas as pd 
import random, math, time, json, random, os, re

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')
logger = logging.getLogger('PythonCodeGen')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Currently running on {device}')



# Break the code into statement and the codes from the cleaned file

file_name = '/content/drive/MyDrive/PythonEnglishDataset/english_python_data_cleaned.txt'


with open(file_name, 'rb') as f:
    f = str(f.read())

f = f.replace('\\n', ' nl ').replace(' ', ' ws ')
pattern = '\#.*?nl'
statement = re.findall(pattern, f)
code = re.split(pattern, f)

statement = [i.replace('ws', '').replace('nl', '').replace('#', '').replace('  ',' ').strip() for i in statement]

del code[0]

final_data = list(zip(statement, code))


def tokenize_en(text):
  return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_code(text):
    return text.split(' ')

STAT = Field(tokenize= tokenize_en, 
            init_token='<sos>', 
            eos_token='<eos>', 
            lower=True,
            batch_first = True)

CODE = Field(tokenize = tokenize_code, 
            init_token='<sos>', 
            eos_token='<eos>', 
            lower=False,
            batch_first = True)


stat_code_pairs = [Example.fromlist([p[0],p[1]], fields) for p in final_data]
stat_code_pairs = Dataset(stat_code_pairs, fields)


fields = [('statement', STAT),('code',CODE)]

train_data, valid_data = stat_code_pairs.split(split_ratio=[0.90,0.10])


STAT.build_vocab(stat_code_pairs, min_freq=2)
CODE.build_vocab(stat_code_pairs, min_freq=2)


BATCH_SIZE = 8

train_iterator, valid_iterator = BucketIterator.splits(
    (train_data, valid_data),
    batch_size = BATCH_SIZE,
    device = device
)