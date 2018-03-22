#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Rank documents with TF-IDF scores"""

import logging
import numpy as np

from multiprocessing.pool import ThreadPool
from functools import partial

from . import utils
from . import DEFAULTS
from .. import tokenizers

logger = logging.getLogger(__name__)


class RLDocRanker(object):
    """
    """

    def __init__(self, model_path=None, tokenizer=None, strict=True):
        """
        Args:
            model_path: path to saved model file
            strict: fail on empty queries or continue (and return empty result)
        """
        # Load from disk
        model_path = model_path or DEFAULTS['rl_path']
        logger.info('Loading %s' % model_path)
        self.tokenizer = tokenizers.get_class(tokenizer or DEFAULTS['tokenizer'])()
        self.num_docs = 1000
        self.strict = strict

    def closest_docs(self, query, k=1):
        """
        """
        doc_scores = [np.random.random() for i in range(k)]
        doc_ids = [np.random.randint(self.num_docs) for i in range(k)]
        return doc_ids, doc_scores

    def batch_closest_docs(self, queries, k=1, num_workers=None):
        """Process a batch of closest_docs requests multithreaded.
        Note: we can use plain threads here as scipy is outside of the GIL.
        """
        with ThreadPool(num_workers) as threads:
            closest_docs = partial(self.closest_docs, k=k)
            results = threads.map(closest_docs, queries)
        return results

    def parse(self, query):
        """Parse the query into tokens (either ngrams or tokens)."""
        tokens = self.tokenizer.tokenize(query)
        return tokens.ngrams(n=self.ngrams, uncased=True,
                             filter_fn=utils.filter_ngram)
