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

from . import utils
from . import DEFAULTS

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Retriever class.
# ------------------------------------------------------------------------------


class Retriever(object):
    """
    """

    def __init__(self, model=None, tokenizer=None,
                 embedding_file=None, num_workers=None):
        """
        Args:
            model: path to saved model file.
            tokenizer: option string to select tokenizer class.
            embedding_file: if provided, will expand dictionary to use all
              available pretrained vectors in this file.
            num_workers: number of CPU processes to use to preprocess batches.
        """
        logger.info('Initializing model...')
        self.model = RLDocRetriever.load(model or DEFAULTS['model'])

        if embedding_file:
            logger.info('Expanding dictionary...')
            words = utils.index_embedding_words(embedding_file)
            added = self.model.expand_dictionary(words)
            self.model.load_embeddings(added, embedding_file)

    def retrieve_docs(self, question, top_n=1):
        """
        Given a question, use search engine to retrive a set of initial documents from a corpus;
        use the terms in these documents as candidate terms to reformulate the question, and perform another search...
        repeat until N rounds...
        use the retrived docs (top k) in the last round as the final result
        """
        results, metrics = self.batch_retrieve_docs([question], top_n)
        return results[0], metrics[0]

    def batch_retrieve_docs(self, questions, top_n=1):
        """Process a batch of retrieve_docs requests multithreaded.
        Note: we can use plain threads here as scipy is outside of the GIL.
        """
        results, metrics = self.model.retrieve(questions, top_n)

        return results, metrics

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
