#!/usr/bin/env python
# encoding: utf-8
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# Adapt from facebookresearch/DrQA by Shuailong on Mar 22 2018.

"""Evaluate the accuracy of the RLQA retriever module."""

import logging
import argparse
import json
import time
import os

from tqdm import tqdm
import torch

from rlqa.retriever import Retriever


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, default=None,
                        help='SQuAD-like dataset to evaluate on')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model to use')
    parser.add_argument('--embedding-file', type=str, default=None,
                        help=('Expand dictionary to use all pretrained '
                              'embeddings in this file.'))
    parser.add_argument('--tokenizer', type=str, default=None,
                        help=("String option specifying tokenizer type to use "
                              "(e.g. 'corenlp')"))
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU processes (for tokenizing, etc)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Use CPU only')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='Specify GPU device id to use')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Example batching size')
    parser.add_argument('--top_n', type=int, default=5,
                        help='retrieve top n docs per question')
    parser.add_argument('--match', type=str, default='string',
                        choices=['regex', 'string'])
    args = parser.parse_args()

    # start time
    start = time.time()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.set_device(args.gpu)
        logger.info('CUDA enabled (GPU %d)' % args.gpu)
    else:
        logger.info('Running on CPU only.')

    # get the closest docs for each question.
    logger.info('Initializing retriever...')
    retriever = Retriever(
        args.model,
        args.tokenizer,
        args.embedding_file,
        args.num_workers,
    )
    if args.cuda:
        retriever.cuda()

    # read all the data and store it
    logger.info('Reading data ...')
    exmaples = []
    for line in open(args.dataset):
        data = json.loads(line)
        exmaples.append((data['question'], data['answers']))

    # compute the scores for each pair, and print the statistics
    logger.info('Retrieving docs and computing scores...')

    scores = []
    for i in tqdm(range(0, len(exmaples), args.batch_size)):
        _, metrics = retriever.batch_retrieve_docs(
            zip(*exmaples[i:i + args.batch_size]), top_n=args.top_n)
        scores.extend([m['hit'] for m in metrics])

    filename = os.path.basename(args.dataset)
    stats = "\n" + "-" * 50 + "\n"
    # f"{filename}\n" +
    # f"Examples:\t\t\t{len(scores)}\n" +
    # f"Matches in top {args.top_n}:\t\t{sum(scores)}\n" +
    # f"Match % in top {args.top_n}:\t\t{(sum(scores) / len(scores) * 100):2.2f}\n" +
    # f"Total time:\t\t\t{time.time() - start:2.4f} (s)\n"

    print(stats)
