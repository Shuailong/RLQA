#!/usr/bin/env python
# encoding: utf-8
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Preprocess the SQuAD dataset for training. Do not use for now."""

import argparse
import os
import sys
import json
import time

from multiprocessing import Pool
from multiprocessing.util import Finalize
from functools import partial
from rlqa import tokenizers

# ------------------------------------------------------------------------------
# Tokenize
# ------------------------------------------------------------------------------

TOK = None


def init(tokenizer_class, options):
    global TOK
    TOK = tokenizer_class(**options)
    Finalize(TOK, TOK.shutdown, exitpriority=100)


def tokenize(text):
    """Call the global process tokenizer on the input text."""
    global TOK
    return TOK.tokenize(text).words()


# ------------------------------------------------------------------------------
# Process dataset examples
# ------------------------------------------------------------------------------


def load_dataset(path):
    """Load json file and store fields separately."""
    with open(path) as f:
        output = [json.loads(line) for line in f]
    return output


def process_dataset(data, tokenizer, workers=None):
    """Iterate processing (tokenize, parse, etc) dataset multithreaded."""
    tokenizer_class = tokenizers.get_class(tokenizer)
    make_pool = partial(Pool, workers, initializer=init)
    workers = make_pool(initargs=(tokenizer_class, {}))

    questions = [qa['question'] for qa in data]
    answers = [qa['answer'] for qa in data]
    q_tokens = workers.map(tokenize, questions)
    workers.close()
    workers.join()

    for idx in range(len(questions)):
        question = q_tokens[idx]
        yield {
            'question': question,
            'answer': answers[idx]
        }


# -----------------------------------------------------------------------------
# Commandline options
# -----------------------------------------------------------------------------


parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str, help='Path to SQuAD data directory')
parser.add_argument('out_dir', type=str, help='Path to output file dir')
parser.add_argument('--split', type=str, help='Filename for train/dev split',
                    default='SQuAD-v1.1-train')
parser.add_argument('--workers', type=int, default=None)
parser.add_argument('--tokenizer', type=str, default='corenlp')
args = parser.parse_args()

t0 = time.time()

in_file = os.path.join(args.data_dir, args.split + '.txt')
print(f'Loading dataset {in_file}', file=sys.stderr)
dataset = load_dataset(in_file)

out_file = os.path.join(
    args.out_dir, f'{args.split}-processed-{args.tokenizer}.txt'
)
print(f'Will write to file {out_file}', file=sys.stderr)
with open(out_file, 'w') as f:
    for ex in process_dataset(dataset, args.tokenizer, args.workers):
        f.write(json.dumps(ex) + '\n')
print(f'Total time: {time.time() - t0:.4f} (s)')
