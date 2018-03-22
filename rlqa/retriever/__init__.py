#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from .. import DATA_DIR

DEFAULTS = {
    'db_path': os.path.join(DATA_DIR, 'wikipedia/docs.db'),
    'rl_path': os.path.join(
        DATA_DIR,
        'rlmodels/rlmodels.model'
    ),
    'tokenizer': 'corenlp'
}


def set_default(key, value):
    global DEFAULTS
    DEFAULTS[key] = value


def get_class(name):
    if name == 'sqlite':
        return DocDB
    if name == 'rl':
        return RLDocRanker
    raise RuntimeError('Invalid retriever class: %s' % name)


from .doc_db import DocDB
from .rl_doc_ranker import RLDocRanker
