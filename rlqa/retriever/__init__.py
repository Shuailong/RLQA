#!/usr/bin/env python
# encoding: utf-8
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from ..tokenizers import CoreNLPTokenizer
from .. import DATA_DIR

DEFAULTS = {
    'db_path': os.path.join(DATA_DIR, 'wikipedia/docs_tokens.db'),
    'model': os.path.join(
        DATA_DIR,
        'rlmodels/best.model'
    ),
    'tokenizer': CoreNLPTokenizer,
    'tfidf_path': os.path.join(
        DATA_DIR,
        'wikipedia/docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz'
    ),
}


def set_default(key, value):
    global DEFAULTS
    DEFAULTS[key] = value


def get_class(name):
    if name == 'sqlite':
        return DocDB
    if name == 'rl':
        return RLDocRetriever
    if name == 'tfidf':
        return
    raise RuntimeError(f'Invalid retriever class: {name}')


from .model import RLDocRetriever
from .retriever import Retriever
from .tfidf_doc_ranker import TfidfDocRanker
from .doc_db import DocDB
from . import config
from . import vector
from . import data
from . import utils
