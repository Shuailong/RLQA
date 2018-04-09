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
import os

from tqdm import tqdm
import numpy as np

import torch

from rlqa.retriever import utils, data, vector
from rlqa.retriever import RLDocRetriever
from rlqa.retriever import DEFAULTS
from rlqa import DATA_DIR as RLQA_DATA


MODEL_DIR = os.path.join(RLQA_DATA, 'rlmodels')

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
                        help='SQuAD-like dataset to evaluate on (txt format)')
    parser.add_argument('--model', type=str, default=None, required=True,
                        help='Path to model to use')
    parser.add_argument('--data-workers', type=int, default=5,
                        help='Number of subprocesses for data loading')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Use CPU only')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='Specify GPU device id to use')
    parser.add_argument('--batch-size', type=int, default=20,
                        help='Example batching size')
    parser.add_argument('--candidate_doc_max', type=int, default=5,
                        help='retrieve top n docs per question')
    parser.add_argument('--ranker_doc_max', type=int, default=5,
                        help='retrieve top n docs per question')
    parser.add_argument('--metrics', type=str, choices=['precision', 'recall', 'F1', 'map', 'hit'],
                        help='metrics to display when training', nargs='+',
                        default=['precision', 'hit'])
    parser.add_argument('--reformulate-rounds', type=int, default=0,
                        help='query reformulate rounds')
    parser.add_argument('--search-engine', type=str, default='lucene', choices=['lucene', 'tfidf_ranker'],
                        help='search engine')
    parser.add_argument('--match', type=str, default='token',
                        choices=['regex', 'string', 'title', 'token'])
    parser.add_argument('--similarity', type=str, default='classic', choices=['classic', 'bm25'],
                        help='lucene search similarity')
    parser.add_argument('--index-folder', type=str, default='index-full-text',
                        help='folder to store lucene\'s index')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.set_device(args.gpu)
        logger.info('CUDA enabled (GPU %d)' % args.gpu)
    else:
        logger.info('Running on CPU only.')

    # --------------------------------------------------------------------------
    # MODEL
    model_path = os.path.join(MODEL_DIR, args.model)
    logger.info('-' * 100)
    model = RLDocRetriever.load(model_path or DEFAULTS['model'], new_args=args)
    if args.cuda and args.reformulate_rounds > 0:
        model.cuda()

    # --------------------------------------------------------------------------
    # DATA
    logger.info('-' * 100)
    logger.info(f'Load data files from {args.dataset}')
    exs = utils.load_data(args.dataset)
    logger.info(f'Num examples = {len(exs)}')

    # --------------------------------------------------------------------------
    # DATA ITERATORS
    # Two datasets: train and dev. If we sort by length it's faster.
    logger.info('-' * 100)
    logger.info('Make data loaders')
    dataset = data.RetriverDataset(exs)

    sampler = torch.utils.data.sampler.SequentialSampler(dataset)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.data_workers,
        collate_fn=vector.batchify,
        pin_memory=args.cuda,
    )

    # -------------------------------------------------------------------------
    # PRINT CONFIG
    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' %
                json.dumps(vars(args), indent=4, sort_keys=True))

    logger.info('MODEL CONFIG:\n%s' %
                json.dumps(vars(model.args), indent=4, sort_keys=True))

    # --------------------------------------------------------------------------
    # EVAL LOOP
    logger.info('-' * 100)
    logger.info('Starting evaluating...')

    eval_time = utils.Timer()
    meters = {k: utils.AverageMeter() for k in args.metrics}

    # Make predictions
    for idx, ex in tqdm(enumerate(loader), total=len(loader)):
        batch_size, metrics = model.retrieve(ex)
        metrics_last = {k: np.mean([m[k] for m in metrics[-1]]).item() for k in args.metrics}
        [meters[k].update(metrics_last[k], batch_size) for k in meters]

    logger.info(' | '.join([f'{k}: {meters[k].avg * 100 :.2f}%' for k in args.metrics]))
