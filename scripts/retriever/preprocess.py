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


def load_json_dataset(path):
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


def process_dataset(data, tokenizer, match):
    processed = []
    for qa in tqdm(data, total=len(data)):
        qa['question'] = tokenizer.tokenize(utils.normalize(qa['question'])).words(uncased=True)
        if match == 'string':
            qa['answer'] = [tokenizer.tokenize(utils.normalize(ans)).words(uncased=True) for ans in qa['answer']]
        else:
            qa['answer'] = [utils.normalize(ans) for ans in qa['answer']]
        processed.append(qa)
    return processed

# -----------------------------------------------------------------------------
# Commandline options
# -----------------------------------------------------------------------------


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


parser = argparse.ArgumentParser()
parser.register('type', 'bool', str2bool)

parser.add_argument('--data_dir', type=str, help='Path to SQuAD data directory',
                    default='data/datasets')
parser.add_argument('--out_dir', type=str, help='Path to output file dir',
                    default='data/datasets')
parser.add_argument('--file', type=str, help='Filename for train/dev split')
parser.add_argument('--tokenizer', type=str, help='tokenizer to tokenize questions',
                    default='corenlp')
parser.add_argument('--match', type=str, default='string', choices=['regex', 'string', 'title'],
                    help='only tokenize answers when match == "string"')

args = parser.parse_args()

in_file = os.path.join(args.data_dir, args.file)
logger.info(f'Loading dataset {in_file}')

if in_file.endswith('.json'):
    dataset = load_json_dataset(in_file)
elif in_file.endswith('.txt'):
    dataset = utils.load_data(in_file)


logger.info(f'Initialize {args.tokenizer} tokenizer...')
tokenizer = tokenizers.get_class(args.tokenizer)()

pairs = process_dataset(dataset, tokenizer, args.match)

basename = os.path.splitext(args.file)[0]
out_file = os.path.join(args.out_dir, f'{basename}-{args.tokenizer}-processed.txt')

with open(out_file, 'w', encoding='utf-8') as f:
    for ex in pairs:
        f.write(json.dumps(ex) + '\n')
logger.info(f'write {len(pairs)} QA pairs to file {out_file}')
