#!/usr/bin/env python
# encoding: utf-8
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Model architecture/optimization options for DrQA document reader."""

import argparse
import logging

logger = logging.getLogger(__name__)

# Index of arguments concerning the core model architecture
MODEL_ARCHITECTURE = {
    'model_type', 'rnn_type', 'embedding_dim', 'hidden_size',
    'question_layers', 'doc_layers'
}

# Index of arguments concerning the model optimizer/training
MODEL_OPTIMIZER = {
    'dropout_emb', 'dropout_rnn', 'dropout_rnn_output', 'optimizer',
    'learning_rate', 'grad_clipping', 'weight_decay', 'momentum',
    'fix_embeddings', 'tune_partial', 'rnn_padding',
    'match', 'reward', 'candidate_term_max', 'candidate_doc_max', 'ranker_doc_max',
    'entropy_regularizer', 'stablize_alpha', 'term_epsilon', 'context_window_size',
    'reformulate_rounds',
    'search_engine', 'num_search_workers',
    'index_folder', 'ngram'
}


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def add_model_args(parser):
    parser.register('type', 'bool', str2bool)

    # Model architecture
    model = parser.add_argument_group('RLQA Retriever Model Architecture')
    model.add_argument('--model-type', type=str, default='rnn',
                       help='Model architecture type')
    model.add_argument('--rnn-type', type=str, default='lstm',
                       help='RNN type: LSTM, GRU, or RNN')
    model.add_argument('--embedding-dim', type=int, default=300,
                       help='Embedding size if embedding_file is not given')
    model.add_argument('--hidden-size', type=int, default=256,
                       help='Hidden size of RNN units')
    model.add_argument('--question-layers', type=int, default=2,
                       help='Number of encoding layers for question')
    model.add_argument('--doc-layers', type=int, default=2,
                       help='Number of encoding layers for document')

    # Optimization details
    optim = parser.add_argument_group('RLQA Retriever Optimization')
    optim.add_argument('--dropout-emb', type=float, default=0.4,
                       help='Dropout rate for word embeddings')
    optim.add_argument('--dropout-rnn', type=float, default=0.4,
                       help='Dropout rate for RNN states')
    optim.add_argument('--dropout-rnn-output', type='bool', default=True,
                       help='Whether to dropout the RNN output')
    optim.add_argument('--optimizer', type=str, default='adam',
                       help='Optimizer: sgd or adamax or adam')
    optim.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate for SGD only')
    optim.add_argument('--grad-clipping', type=float, default=1,
                       help='Gradient clipping')
    optim.add_argument('--weight-decay', type=float, default=0,
                       help='Weight decay factor')
    optim.add_argument('--momentum', type=float, default=0,
                       help='Momentum factor')
    optim.add_argument('--fix-embeddings', type='bool', default=True,
                       help='Keep word embeddings fixed (use pretrained)')
    optim.add_argument('--tune-partial', type=int, default=0,
                       help='Backprop through only the top N question words')
    optim.add_argument('--rnn-padding', type='bool', default=False,
                       help='Explicitly account for padding in RNN encoding')

    # RL Training hyperparams
    rl_params = parser.add_argument_group('RLQA Retriever Doc Selection')
    rl_params.add_argument('--match', type=str, default='string',
                           choices=['regex', 'string', 'title', 'token'])
    rl_params.add_argument('--reward', type=str, default='precision',
                           choices=['precision', 'recall', 'F1', 'map', 'hit', 'hit@5'],
                           help='reward to train the reformulator')
    rl_params.add_argument('--candidate-term-max', type=int, default=300,
                           help='First M words to select from the candidate doc')
    rl_params.add_argument('--candidate-doc-max', type=int, default=5,
                           help='First K docs to select as candidate doc')
    rl_params.add_argument('--ranker-doc-max', type=int, default=40,
                           help='First k docs to returned by ranker')
    rl_params.add_argument('--entropy-regularizer', type=float, default=2e-3,
                           help='Cross entropy regularization coefficient lambda')
    rl_params.add_argument('--stablize-alpha', type=float, default=0.1,
                           help='Value loss coefficient alpha')
    rl_params.add_argument('--term-epsilon', type=float, default=0.5,
                           help='Threshold value to select terms')
    rl_params.add_argument('--context-window-size', type=int, default=5,
                           help='context window size for candidate terms. should be odd number.')
    rl_params.add_argument('--reformulate-rounds', type=int, default=1,
                           help='query reformulate rounds')

    # Search Engine settings
    search = parser.add_argument_group('RLQA Retriever Search Engine')
    search.add_argument('--search-engine', type=str, default='lucene', choices=['lucene', 'tfidf_ranker'],
                        help='search engine')
    search.add_argument('--num-search-workers', type=int, default=20,
                        help='search engine workers')
    search.add_argument('--index-folder', type=str, default='index',
                        help='folder to store lucene\'s index')
    search.add_argument('--ngram', type=int, default=1, help='ngram to compute similarity')


def get_model_args(args):
    """Filter args for model ones.

    From a args Namespace, return a new Namespace with *only* the args specific
    to the model architecture or optimization. (i.e. the ones defined here.)
    """
    global MODEL_ARCHITECTURE, MODEL_OPTIMIZER
    required_args = MODEL_ARCHITECTURE | MODEL_OPTIMIZER
    arg_values = {k: v for k, v in vars(args).items() if k in required_args}
    return argparse.Namespace(**arg_values)


def override_model_args(old_args, new_args):
    """Set args to new parameters.

    Decide which model args to keep and which to override when resolving a set
    of saved args and new args.

    We keep the new optimation, but leave the model architecture alone.
    """
    global MODEL_OPTIMIZER
    old_args, new_args = vars(old_args), vars(new_args)
    for k in old_args.keys():
        if k in new_args and old_args[k] != new_args[k]:
            if k in MODEL_OPTIMIZER:
                logger.info('Overriding saved %s: %s --> %s' %
                            (k, old_args[k], new_args[k]))
                old_args[k] = new_args[k]
            else:
                logger.info('Keeping saved %s: %s' % (k, old_args[k]))
    return argparse.Namespace(**old_args)
