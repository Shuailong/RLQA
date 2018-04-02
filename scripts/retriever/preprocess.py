#!/usr/bin/env python
# encoding: utf-8
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Preprocess the SQuAD dataset for training. Remove invalid queries."""

import argparse
import os
import json
import logging

from tqdm import tqdm

from rlqa.retriever import TfidfDocRanker
from rlqa.retriever import utils


logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


def is_valid(query, ranker):
    """Call the global process tokenizer on the input text."""
    words = ranker.parse(utils.normalize(query))
    wids = [utils.hash(w, ranker.hash_size) for w in words]

    return len(wids) != 0


def load_dataset(path):
    """Load json file and store fields separately."""
    with open(path) as f:
        output = [json.loads(line) for line in f]
    return output


def process_dataset(data, ranker):
    valids, invalids = [], []
    for qa in data:
        q, a = qa['question'], qa['answer']
        if is_valid(q, ranker):
            valids.append(qa)
        else:
            invalids.append(qa)
    return valids, invalids

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
args = parser.parse_args()

in_file = os.path.join(args.data_dir, args.split + '.txt')
logger.info(f'Loading dataset {in_file}')
dataset = load_dataset(in_file)

valid_file = os.path.join(args.out_dir, f'{args.split}-valid.txt')
invalid_file = os.path.join(args.out_dir, f'{args.split}-invalid.txt')

logger.info('Initialize ranker...')
ranker = TfidfDocRanker(strict=True)
valids, invalids = process_dataset(dataset, ranker)

with open(valid_file, 'w', encoding='utf-8') as f:
    for ex in tqdm(valids, total=len(valids)):
        f.write(json.dumps(ex) + '\n')
logger.info(f'write {len(valids)} QA pairs to file {valid_file}')

with open(invalid_file, 'w', encoding='utf-8') as f:
    for ex in tqdm(invalids, total=len(invalids)):
        f.write(json.dumps(ex) + '\n')
logger.info(f'write {len(invalids)} QA pairs to file {invalid_file}')
