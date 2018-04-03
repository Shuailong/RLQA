#!/usr/bin/env python
# encoding: utf-8
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Preprocess the SQuAD dataset for training. Tokenize questions. Remove invalid queries(TFIDF ranker). Match docs"""

import argparse
import os
import json
import logging

from tqdm import tqdm
from rlqa.retriever import utils
from rlqa import tokenizers


logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


def is_valid(query, tokenizer):
    """Call the global process tokenizer on the input text."""
    NGRAM = 2
    HASH_SIZE = 16777216
    tokens = tokenizer.tokenize(utils.normalize(query))
    words = tokens.ngrams(n=NGRAM, uncased=True,
                          filter_fn=utils.filter_ngram)

    wids = [utils.hash(w, HASH_SIZE) for w in words]

    return len(wids) != 0


def load_dataset(path):
    """Load json file and store fields separately."""
    # Read dataset
    data = []
    with open(path, encoding='utf-8') as f:
        dataset = json.load(f)
    # Iterate and write question-answer pairs
    for article in dataset['data']:
        doc_truth = [utils.normalize(article['title'].replace('_', ' '))]
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                question = qa['question']
                answer = [a['text'] for a in qa['answers']]
                data.append({'question': question, 'answer': answer, 'doc_truth': doc_truth})
    return data


def process_dataset(data, tokenizer):
    valids = []
    for qa in tqdm(data, total=len(data)):
        if is_valid(qa['question'], tokenizer):
            qa['question'] = tokenizer.tokenize(qa['question']).words(uncased=True)
            valids.append(qa)
        else:
            logger.warning(f'WARN: invalid question: {qa["question"]}.')
    return valids


# -----------------------------------------------------------------------------
# Commandline options
# -----------------------------------------------------------------------------


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='Path to SQuAD data directory',
                    default='data/datasets')
parser.add_argument('--out_dir', type=str, help='Path to output file dir',
                    default='data/datasets')
parser.add_argument('--split', type=str, help='Filename for train/dev split',
                    default='SQuAD-v1.1-train')

parser.add_argument('--tokenizer', type=str, help='tokenizer to tokenize questions',
                    default='corenlp')
args = parser.parse_args()

in_file = os.path.join(args.data_dir, args.split + '.json')
logger.info(f'Loading dataset {in_file}')
dataset = load_dataset(in_file)

logger.info(f'Initialize {args.tokenizer} tokenizer...')
tokenizer = tokenizers.get_class(args.tokenizer)()
pairs = process_dataset(dataset, tokenizer)

out_file = os.path.join(args.out_dir, f'{args.split}-{args.tokenizer}-processed.txt')

with open(out_file, 'w', encoding='utf-8') as f:
    for ex in pairs:
        f.write(json.dumps(ex) + '\n')
logger.info(f'write {len(pairs)} QA pairs to file {out_file}')
