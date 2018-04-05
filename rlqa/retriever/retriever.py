#!/usr/bin/env python
# encoding: utf-8
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Retrieve relevant documents with using reformulated queries"""

import logging

from .model import RLDocRetriever

from . import DEFAULTS

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Retriever class.
# ------------------------------------------------------------------------------


class Retriever(object):
    """
    """

    def __init__(self, model=None, num_workers=None):
        """
        Args:
            model: path to saved model file.
            num_workers: number of CPU processes to use to preprocess batches.
        """
        logger.info('Initializing model...')
        self.model = RLDocRetriever.load(model or DEFAULTS['model'])

    def retrieve_docs(self, question, top_n=1):
        """
        Given a question, use search engine to retrive a set of initial documents from a corpus;
        use the terms in these documents as candidate terms to reformulate the question, and perform another search...
        repeat until N rounds...
        use the retrived docs (top k) in the last round as the final result
        """
        docs_pred, questions, probs, selections, reward_baseline = self.model.retrieve([question])

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
