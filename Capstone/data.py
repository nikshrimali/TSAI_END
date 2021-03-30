# data.py - Contains code to load and extract the data

import spacy
spacy_en = spacy.load("en_core_web_sm")

import numpy as np
import pandas as pd 
import random
import os
import re

from torchtext.legacy.data import Field, BucketIterator, TabularDataset, Example, Dataset


class GetData:
    def __init__(self, file_name):
        self.file_name = os.path.join(os.getcwd(), file_name)

    def readdata(self):
        
        with open(self.file_name, 'rb') as f:
            f = str(f.read())

        f = f.replace('\\n', ' nl ').replace(' ', ' ws ')
        pattern = '\#.*?nl'
        statement = re.findall(pattern, f)
        code = re.split(pattern, f)
        statement = [i.replace('ws', '').replace('nl', '').replace('#', '').replace('  ',' ').strip() for i in statement]
        del code[0]
        return list(zip(statement, code))

    @staticmethod
    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    @staticmethod
    def tokenize_code(text):
        return text.split(' ')

    def load_iterator(self, batch_size:int, device='cuda'):

        STAT = Field(tokenize= self.tokenize_en, 
                    init_token='<sos>', 
                    eos_token='<eos>', 
                    lower=True,
                    batch_first = True)

        CODE = Field(tokenize = self.tokenize_code, 
                    init_token='<sos>', 
                    eos_token='<eos>', 
                    lower=False,
                    batch_first = True)
        
        final_data = self.readdata()

        fields = [('code', CODE), ('statement', STAT)]

        stat_code_pairs = [Example.fromlist([p[0],p[1]], fields) for p in final_data]
        stat_code_pairs = Dataset(stat_code_pairs, fields)



        train_data, valid_data = stat_code_pairs.split(split_ratio=[0.90,0.10])
        # print('Train_data is ', vars(train_data))
        # print('Valid_data is ', vars(valid_data))
        
        # return train_data, valid_data

        CODE.build_vocab(stat_code_pairs, min_freq=2)
        STAT.build_vocab(stat_code_pairs, min_freq=2)

        # train_iterator, valid_iterator = BucketIterator.splits(
        #     (train_data, valid_data),
        #     batch_size = batch_size,
        #     device = device
        # )

        train_iterator, valid_iterator = BucketIterator.splits(
            (train_data, valid_data), 
            batch_size = batch_size,
            sort_key = lambda x: len(x.statement),
            sort_within_batch = True,
            device = device)
        return STAT, CODE, train_iterator, valid_iterator