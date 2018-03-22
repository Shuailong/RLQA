#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# Adapt from facebookresearch/DrQA by Shuailong on Mar 22 2018.

"""A script to convert the default WebQuestions dataset to the format:

'{"question": "q1", "answer": ["a11", ..., "a1i"]}'
...
'{"question": "qN", "answer": ["aN1", ..., "aNi"]}'

"""

import argparse
import re
import json

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('output', type=str)
args = parser.parse_args()

# Read dataset
with open(args.input) as f:
    dataset = json.load(f)

# Iterate and write question-answer pairs
with open(args.output, 'w') as f:
    for ex in dataset:
        question = ex['utterance']
        answer = ex['targetValue']
        answer = re.findall(
            r'(?<=\(description )(.+?)(?=\) \(description|\)\)$)', answer
        )
        answer = [a.replace('"', '') for a in answer]
        f.write(json.dumps({'question': question, 'answer': answer}))
        f.write('\n')
