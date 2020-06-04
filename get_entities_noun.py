"""
For hotpot Pipeline in codalab
Rewrite eval.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from NER_model import Net
from data_load import NerDataset, pad, VOCAB, tokenizer, tag2idx, idx2tag, EvalDataset, QueryDataset
import os
import numpy as np
import argparse
from tqdm import tqdm
import json
import re
import spacy
nlp = spacy.load('en')



def get_entities(words):
    doc = nlp(words)
    # entities = set()
    ents = doc.ents
    # nouns = list(doc.noun_chunks)
    # for ent in ents:
    #     entities.add(ent.text)
    # for noun in nouns:
    #     entities.add(noun.text)
    return list(map(str, ents))


def eval_query(dataset, output_path):
    entities = dict()
    for case in tqdm(dataset):
        key = case['_id']
        sent = case['question']
        entity = get_entities(sent)
        entities[key] = entity
    json.dump(entities, open(output_path, 'w', encoding='utf-8'))
    return


def eval_para(dataset, output_path):
    entities = dict()
    for case in tqdm(dataset):
        key = case['_id']
        paras = case['context']
        tmp_entities = dict()
        for para in paras:
            title = para[0]
            sents = para[1]
            # para_entities = dict()
            tmp_sents = ""
            for sent_id, sent in enumerate(sents):
                tmp_sents += sent
                tmp_sents += " "
                # para_entities[sent_id] = get_entities(sent)
            para_entities = get_entities(tmp_sents)
            tmp_entities[title] = para_entities

        entities[key] = tmp_entities
    json.dump(entities, open(output_path, 'w'))
    return


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default=r'E:\Project\hotpot\bert_ner/bert_ner.pt')
    parser.add_argument('--input_path', type=str, default=r"E:\DATA\HotpotQA\sample_examples.json")
    parser.add_argument('--input_query_path', type=str, default=r"E:\DATA/HotpotQA/hotpot_dev_distractor_v1.json")
    parser.add_argument('--output_path', type=str, default=r'E:\DATA\HotpotQA\entities/dev_entities.json')
    parser.add_argument('--output_query_path', type=str, default=r'E:\DATA\HotpotQA\entities/dev_query_entities.json')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--use_query', action='store_true')
    args = parser.parse_args()

    data = json.load(open(args.input_query_path, 'r'))

    eval_query(data, args.output_query_path)

    eval_para(data, args.output_path)
