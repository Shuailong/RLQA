# !/usr/bin/env python
# encoding: utf-8
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Main RLQA retriever training script."""

import argparse
import torch
import numpy as np
import json
import os
import sys
import subprocess
import logging

from rlqa.retriever import utils, vector, config, data
from rlqa.retriever import RLDocRetriever
from rlqa import tokenizers
from rlqa import DATA_DIR as RLQA_DATA


logger = logging.getLogger()


# ------------------------------------------------------------------------------
# Training arguments.
# ------------------------------------------------------------------------------


# Defaults
DATA_DIR = os.path.join(RLQA_DATA, 'datasets')
MODEL_DIR = os.path.join(RLQA_DATA, 'rlmodels')
EMBED_DIR = os.path.join(RLQA_DATA, 'embeddings')


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def add_train_args(parser):
    """Adds commandline arguments pertaining to training a model. These
    are different from the arguments dictating the model architecture.
    """
    parser.register('type', 'bool', str2bool)

    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--no-cuda', type='bool', default=False,
                         help='Train on CPU, even if GPUs are available.')
    runtime.add_argument('--gpu', type=int, default=-1,
                         help='Run on a specific GPU')
    runtime.add_argument('--data-workers', type=int, default=5,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--parallel', type='bool', default=False,
                         help='Use DataParallel on all available GPUs')
    runtime.add_argument('--random-seed', type=int, default=712,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--num-epochs', type=int, default=100,
                         help='Train data iterations')
    runtime.add_argument('--batch-size', type=int, default=10,
                         help='Batch size for training')
    runtime.add_argument('--test-batch-size', type=int, default=10,
                         help='Batch size during validation/testing')

    # Files
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--model-dir', type=str, default=MODEL_DIR,
                       help='Directory for saved models/checkpoints/logs')
    files.add_argument('--model-name', type=str, default='',
                       help='Unique model identifier (.mdl, .txt, .checkpoint)')
    files.add_argument('--data-dir', type=str, default=DATA_DIR,
                       help='Directory of training/validation data')
    files.add_argument('--train-file', type=str,
                       default='SQuAD-v1.1-train-valid.txt',
                       help='train file')
    files.add_argument('--dev-file', type=str,
                       default='SQuAD-v1.1-dev.txt',
                       help='dev file')
    files.add_argument('--embed-dir', type=str, default=EMBED_DIR,
                       help='Directory of pre-trained embedding files')
    files.add_argument('--embedding-file', type=str,
                       default='glove.6B.300d.txt',
                       help='Space-separated pretrained embeddings file')

    # Saving + loading
    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument('--checkpoint', type='bool', default=False,
                           help='Save model + optimizer state after each epoch')
    save_load.add_argument('--pretrained', type=str, default='',
                           help='Path to a pretrained model to warm-start with')
    save_load.add_argument('--expand-dictionary', type='bool', default=False,
                           help='Expand dictionary of pretrained model to ' +
                                'include training/dev words of new data')
    # Data preprocessing
    preprocess = parser.add_argument_group('Preprocessing')
    preprocess.add_argument('--uncased-question', type='bool', default=True,
                            help='Question words will be lower-cased')
    preprocess.add_argument('--uncased-doc', type='bool', default=True,
                            help='Document words will be lower-cased')
    preprocess.add_argument('--restrict-vocab', type='bool', default=False,
                            help='Only use pre-trained words in embedding_file')
    preprocess.add_argument('--restrict-vocab-size', type=int, default=None,
                            help='Only use this number of external vocabulary')
    preprocess.add_argument('--tokenizer', type=str, default='corenlp')

    # General
    general = parser.add_argument_group('General')
    general.add_argument('--valid-metric', type=str, default='precision',
                         help='The evaluation metric used for model selection')
    general.add_argument('--display-iter', type=int, default=25,
                         help='Log state after every <display_iter> epochs')
    general.add_argument('--sort-by-len', type='bool', default=True,
                         help='Sort batches by length for speed')


def set_defaults(args):
    """Make sure the commandline arguments are initialized properly."""
    # Check critical files exist
    args.train_file = os.path.join(args.data_dir, args.train_file)
    if not os.path.isfile(args.train_file):
        raise IOError('No such file: %s' % args.train_file)
    args.dev_file = os.path.join(args.data_dir, args.dev_file)
    if not os.path.isfile(args.dev_file):
        raise IOError('No such file: %s' % args.dev_file)
    if args.embedding_file:
        args.embedding_file = os.path.join(args.embed_dir, args.embedding_file)
        if not os.path.isfile(args.embedding_file):
            raise IOError('No such file: %s' % args.embedding_file)

    # Set model directory
    subprocess.call(['mkdir', '-p', args.model_dir])

    # Set model name
    if not args.model_name:
        import uuid
        import time
        args.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

    # Set log + model file names
    args.log_file = os.path.join(args.model_dir, args.model_name + '.txt')
    args.model_file = os.path.join(args.model_dir, args.model_name + '.mdl')

    # Embeddings options
    if args.embedding_file:
        with open(args.embedding_file) as f:
            dim = len(f.readline().strip().split(' ')) - 1
        args.embedding_dim = dim
    elif not args.embedding_dim:
        raise RuntimeError('Either embedding_file or embedding_dim '
                           'needs to be specified.')

    # Make sure tune_partial and fix_embeddings are consistent.
    if args.tune_partial > 0 and args.fix_embeddings:
        logger.warning('WARN: fix_embeddings set to False as tune_partial > 0.')
        args.fix_embeddings = False

    # Make sure fix_embeddings and embedding_file are consistent
    if args.fix_embeddings:
        if not (args.embedding_file or args.pretrained):
            logger.warning('WARN: fix_embeddings set to False '
                           'as embeddings are random.')
            args.fix_embeddings = False

    # Make sure ranker return more docs than candidate docs to return to the reformulator
    if args.ranker_doc_max < args.candidate_doc_max:
        raise RuntimeError('ranker-doc-max >= candidate-doc-max.')

    return args


# ------------------------------------------------------------------------------
# Initalization from scratch.
# ------------------------------------------------------------------------------


def init_from_scratch(args, tokenizer, train_exs, dev_exs):
    """New model, new data, new dictionary."""

    # Build a dictionary from the data questions + words (train/dev splits)
    logger.info('-' * 100)
    logger.info('Build dictionary')
    word_dict = utils.build_word_dict(args, tokenizer, train_exs + dev_exs)
    logger.info(f'Num words in dataset = {len(word_dict)}')

    # Initialize model
    model = RLDocRetriever(config.get_model_args(args), word_dict)

    # Load pretrained embeddings for words in dictionary
    if args.embedding_file:
        words = utils.index_embedding_words(args.embedding_file)
        model.expand_dictionary(words)
        model.load_embeddings(model.word_dict, args.embedding_file)

    return model


# ------------------------------------------------------------------------------
# Train loop.
# ------------------------------------------------------------------------------

def train(args, data_loader, model, global_stats):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    train_loss = utils.AverageMeter()
    epoch_time = utils.Timer()

    # Run one epoch
    for idx, ex in enumerate(data_loader):
        train_loss.update(*model.update(ex))
        if idx % args.display_iter == 0:
            logger.info(f'train: Epoch = {global_stats["epoch"]} | iter = {idx}/{ len(data_loader)} | ' +
                        f'loss = {train_loss.avg:.2f} | elapsed time = {global_stats["timer"].time():.2f} (s)')
            train_loss.reset()

    logger.info(f'train: Epoch {global_stats["epoch"]} done. Time for epoch = {epoch_time.time():.2f} (s)')

    # Checkpoint
    if args.checkpoint:
        model.checkpoint(args.model_file + '.checkpoint',
                         global_stats['epoch'] + 1)


# ------------------------------------------------------------------------------
# Validation loops. Includes functions that
# use different metrics and implementations.
# ------------------------------------------------------------------------------


def validate(args, data_loader, model, global_stats, mode):
    """Run one full validation.
    """
    eval_time = utils.Timer()
    precision = utils.AverageMeter()

    # Make predictions
    examples = 0
    for idx, ex in enumerate(data_loader):
        batch_size = len(ex[0])
        results, metrics = model.retrieve(ex)
        metrics_avg = {k: np.mean([m[k] for m in metrics]).item() for k in metrics[0].keys()}
        precision.update(metrics_avg['precision'], batch_size)
        # If getting train accuracies, sample max 10k
        examples += batch_size
        if mode == 'train' and examples >= 1e4:
            break

    logger.info(f'{mode} valid: Epoch = {global_stats["epoch"]} | precision = {precision.avg:.2f} | ' +
                f'examples = {examples} | valid time = {eval_time.time():.2f} (s)')

    return {'precision': precision.avg}


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


def main(args):
    # --------------------------------------------------------------------------
    # DATA
    logger.info('-' * 100)
    logger.info('Load data files')
    train_exs = utils.load_data(args.train_file)
    logger.info('Num train examples = %d' % len(train_exs))
    dev_exs = utils.load_data(args.dev_file)
    logger.info('Num dev examples = %d' % len(dev_exs))

    # --------------------------------------------------------------------------
    # MODEL
    logger.info('-' * 100)
    start_epoch = 0
    if args.checkpoint and os.path.isfile(args.model_file + '.checkpoint'):
        # Just resume training, no modifications.
        logger.info('Found a checkpoint...')
        checkpoint_file = args.model_file + '.checkpoint'
        model, start_epoch = RLDocRetriever.load_checkpoint(checkpoint_file, args)
    else:
        # Training starts fresh. But the model state is either pretrained or
        # newly (randomly) initialized.
        tokenizer = tokenizers.get_class(args.tokenizer)()
        if args.pretrained:
            logger.info('Using pretrained model...')
            model = RLDocRetriever.load(args.pretrained, args)
            if args.expand_dictionary:
                logger.info('Expanding dictionary for new data...')
                # Add words in training + dev examples
                words = utils.load_words(args, tokenizer, train_exs + dev_exs)
                added = model.expand_dictionary(words)
                # Load pretrained embeddings for added words
                if args.embedding_file:
                    model.load_embeddings(added, args.embedding_file)

        else:
            logger.info('Training model from scratch...')
            model = init_from_scratch(args, tokenizer, train_exs, dev_exs)

        # Set up partial tuning of embeddings
        if args.tune_partial > 0:
            logger.info('-' * 100)
            logger.info('Counting %d most frequent question words' %
                        args.tune_partial)
            top_words = utils.top_question_words(
                args, tokenizer, train_exs, model.word_dict
            )
            for word in top_words[:5]:
                logger.info(word)
            logger.info('...')
            for word in top_words[-6:-1]:
                logger.info(word)
            model.tune_embeddings([w[0] for w in top_words])

        # Set up optimizer
        model.init_optimizer()

    # Use the GPU?
    if args.cuda:
        model.cuda()

    # Use multiple GPUs?
    if args.parallel:
        model.parallelize()

    # --------------------------------------------------------------------------
    # DATA ITERATORS
    # Two datasets: train and dev. If we sort by length it's faster.
    logger.info('-' * 100)
    logger.info('Make data loaders')
    train_dataset = data.RetriverDataset(train_exs)
    if args.sort_by_len:
        train_sampler = data.SortedBatchSampler(train_dataset.lengths(),
                                                args.batch_size,
                                                shuffle=True)
    else:
        # train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
        train_sampler = torch.utils.data.sampler.SequentialSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.data_workers,
        collate_fn=vector.batchify,
        pin_memory=args.cuda,
    )
    dev_dataset = data.RetriverDataset(dev_exs)
    if args.sort_by_len:
        dev_sampler = data.SortedBatchSampler(dev_dataset.lengths(),
                                              args.test_batch_size,
                                              shuffle=False)
    else:
        dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.test_batch_size,
        sampler=dev_sampler,
        num_workers=args.data_workers,
        collate_fn=vector.batchify,
        pin_memory=args.cuda,
    )

    # -------------------------------------------------------------------------
    # PRINT CONFIG
    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' %
                json.dumps(vars(args), indent=4, sort_keys=True))

    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    logger.info('-' * 100)
    logger.info('Starting training...')
    stats = {'timer': utils.Timer(), 'epoch': 0, 'best_valid': 0}
    for epoch in range(start_epoch, args.num_epochs):
        stats['epoch'] = epoch

        # Train
        train(args, train_loader, model, stats)

        # Validate train
        validate(args, train_loader, model, stats, mode='train')

        # Validate dev
        result = validate(args, dev_loader, model, stats, mode='dev')

        # Save best valid
        if result[args.valid_metric] > stats['best_valid']:
            logger.info(f'Best valid: {args.valid_metric} = {result[args.valid_metric]:.2f} ' +
                        f'(epoch {stats["epoch"]}, {model.updates} updates)')
            model.save(args.model_file)
            stats['best_valid'] = result[args.valid_metric]


if __name__ == '__main__':
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        'RLQA Document Retriever',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_train_args(parser)
    config.add_model_args(parser)
    args = parser.parse_args()
    set_defaults(args)

    # Set cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.set_device(args.gpu)

    # Set random state
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Set logging
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        if args.checkpoint:
            logfile = logging.FileHandler(args.log_file, 'a')
        else:
            logfile = logging.FileHandler(args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))

    # Run!
    main(args)
