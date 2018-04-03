#!/usr/bin/env python
# encoding: utf-8
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
Model code.
'''


import logging
import copy
import itertools

import numpy as np
from termcolor import colored

import torch
from torch.autograd import Variable
import torch.optim as optim

from .config import override_model_args
from .reformulator import Reformulator
from .tfidf_doc_ranker import TfidfDocRanker
from .lucene_search import LuceneSearch
from .utils import retrieve_metrics


"""RLQA Document Retrival model"""

logger = logging.getLogger(__name__)


class RLDocRetriever(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(self, args, word_dict, state_dict=None):
        # Book-keeping.
        self.args = args
        self.word_dict = word_dict
        self.args.vocab_size = len(word_dict)
        self.updates = 0
        self.use_cuda = False
        self.parallel = False

        # Building network.
        if args.model_type == 'rnn':
            self.network = Reformulator(args)
        else:
            raise RuntimeError(f'Unsupported model: {args.model_type}')

        if args.search_engine == 'lucene':
            self.search_engine = LuceneSearch(args, word_dict)
        else:
            self.search_engine = TfidfDocRanker(args, word_dict)
        # Load saved state
        if state_dict:
            # Load buffer separately
            if 'fixed_embedding' in state_dict:
                fixed_embedding = state_dict.pop('fixed_embedding')
                self.network.load_state_dict(state_dict)
                self.network.register_buffer(
                    'fixed_embedding', fixed_embedding)
            else:
                self.network.load_state_dict(state_dict)

    def expand_dictionary(self, words):
        """Add words to the DocReader dictionary if they do not exist. The
        underlying embedding matrix is also expanded (with random embeddings).

        Args:
            words: iterable of tokens to add to the dictionary.
        Output:
            added: set of tokens that were added.
        """
        to_add = {self.word_dict.normalize(w) for w in words
                  if w not in self.word_dict}

        # Add words to dictionary and expand embedding layer
        if len(to_add) > 0:
            logger.info('Adding %d new words to dictionary...' % len(to_add))
            for w in to_add:
                self.word_dict.add(w)
            self.args.vocab_size = len(self.word_dict)
            logger.info('New vocab size: %d' % len(self.word_dict))

            old_embedding = self.network.embedding.weight.data
            self.network.embedding = torch.nn.Embedding(self.args.vocab_size,
                                                        self.args.embedding_dim,
                                                        padding_idx=0)
            new_embedding = self.network.embedding.weight.data
            new_embedding[:old_embedding.size(0)] = old_embedding

        # Return added words
        return to_add

    def load_embeddings(self, words, embedding_file):
        """Load pretrained embeddings for a given list of words, if they exist.

        Args:
            words: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        words = {w for w in words if w in self.word_dict}
        logger.info(f'Loading pre-trained embeddings for {len(words)} words from {embedding_file}')
        embedding = self.network.embedding.weight.data

        # When normalized, some words are duplicated. (Average the embeddings).
        vec_counts = {}
        with open(embedding_file, encoding='utf-8') as f:
            for line in f:
                parsed = line.rstrip().split(' ')
                assert(len(parsed) == embedding.size(1) + 1)
                w = self.word_dict.normalize(parsed[0])
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[self.word_dict[w]].copy_(vec)
                    else:
                        logging.warning(f'WARN: Duplicate embedding found for {w.encode("utf-8")}')
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[self.word_dict[w]].add_(vec)

        for w, c in vec_counts.items():
            embedding[self.word_dict[w]].div_(c)

        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))

    def tune_embeddings(self, words):
        """Unfix the embeddings of a list of words. This is only relevant if
        only some of the embeddings are being tuned (tune_partial = N).

        Shuffles the N specified words to the front of the dictionary, and saves
        the original vectors of the other N + 1:vocab words in a fixed buffer.

        Args:
            words: iterable of tokens contained in dictionary.
        """
        words = {w for w in words if w in self.word_dict}

        if len(words) == 0:
            logger.warning('Tried to tune embeddings, but no words given!')
            return

        if len(words) == len(self.word_dict):
            logger.warning('Tuning ALL embeddings in dictionary')
            return

        # Shuffle words and vectors
        embedding = self.network.embedding.weight.data
        for idx, swap_word in enumerate(words, self.word_dict.START):
            # Get current word + embedding for this index
            curr_word = self.word_dict[idx]
            curr_emb = embedding[idx].clone()
            old_idx = self.word_dict[swap_word]

            # Swap embeddings + dictionary indices
            embedding[idx].copy_(embedding[old_idx])
            embedding[old_idx].copy_(curr_emb)
            self.word_dict[swap_word] = idx
            self.word_dict[idx] = swap_word
            self.word_dict[curr_word] = old_idx
            self.word_dict[old_idx] = curr_word

        # Save the original, fixed embeddings
        self.network.register_buffer(
            'fixed_embedding', embedding[idx + 1:].clone()
        )

    def init_optimizer(self, state_dict=None):
        """Initialize an optimizer for the free parameters of the network.

        Args:
            state_dict: network parameters
        """
        if self.args.fix_embeddings:
            for p in self.network.embedding.parameters():
                p.requires_grad = False
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, self.args.learning_rate,
                                       momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(parameters,
                                        weight_decay=self.args.weight_decay)
        else:
            raise RuntimeError(f'Unsupported optimizer: {self.args.optimizer}')

    # --------------------------------------------------------------------------
    # Learning
    # --------------------------------------------------------------------------

    def update(self, ex):
        """Forward a batch of examples; step the optimizer to update weights."""
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

        # Train mode
        self.network.train()

        questions, docs_truth = ex
        # questions: [[word]]

        # vectorize questions
        questions_tensors = [torch.LongTensor([self.word_dict[w] for w in q]) for q in questions]
        # Batch questions
        max_q_length = max([q.size(0) for q in questions_tensors])
        x1 = torch.LongTensor(len(questions_tensors), max_q_length).zero_()
        x1_mask = torch.ByteTensor(len(questions_tensors), max_q_length).fill_(1)
        for i, q in enumerate(questions_tensors):
            x1[i, :q.size(0)].copy_(q)
            x1_mask[i, :q.size(0)].fill_(0)

        # retrieve docs
        question_str = [' '.join(q) for q in questions]
        results = self.search_engine.batch_closest_docs(
            question_str, ranker_doc_max=self.args.ranker_doc_max, candidate_doc_max=self.args.candidate_doc_max)

        docs_pred, term_idxs, terms = zip(*results)

        rewards = [retrieve_metrics(doc_truth, doc_pred)
                   for (doc_truth, doc_pred) in zip(docs_truth, docs_pred)]

        for i in range(len(docs_truth)):
            if np.mean([r['hit'] for r in rewards]).item() == 0:
                logger.debug(colored(f'{" , ".join(docs_truth[i])}', color='yellow') + ' | ' + f'{" , ".join(docs_pred[i])}')


        rewards_recall = [r['recall'] for r in rewards]

        # train time: sample a doc
        term_idxs = [term_idx[0] for term_idx in term_idxs]
        candidate_terms = [term[0] for term in terms]

        # pad head and tail with context window size
        pad_size = (self.args.context_window_size - 1) // 2

        term_idxs_pad = [[0] * pad_size + d + [0] * pad_size for d in term_idxs]
        term_idxs_ctxs = [[d[i - pad_size:i + pad_size + 1]
                           for i in range(len(d)) if i >= pad_size and i <= len(d) - pad_size - 1]
                          for d in term_idxs_pad]
        # term_idxs_ctxs: batch * terms * win_size
        max_terms_length = max([len(doc_terms) for doc_terms in term_idxs_ctxs])
        x2 = torch.LongTensor(len(term_idxs_ctxs), max_terms_length, self.args.context_window_size).zero_()
        x2_mask = torch.ByteTensor(len(term_idxs_ctxs), max_terms_length).fill_(1)

        for i, doc_terms in enumerate(term_idxs_ctxs):
            x2_mask[i, :len(doc_terms)].fill_(0)
            for j, term_contexts in enumerate(doc_terms):
                for k, word in enumerate(term_contexts):
                    x2[i, j, k] = term_idxs_ctxs[i][j][k]

        # Transfer to GPU
        if self.use_cuda:
            x1 = Variable(x1.cuda(async=True))
            x1_mask = Variable(x1_mask.cuda(async=True))
            x2 = Variable(x2.cuda(async=True))
            x2_mask = Variable(x2_mask.cuda(async=True))
        else:
            x1 = Variable(x1)
            x1_mask = Variable(x1_mask)
            x2 = Variable(x2)
            x2_mask = Variable(x2_mask)

        # Run forward
        probs, reward_baseline = self.network(x1, x1_mask, x2, x2_mask)
        # probs: batch * len
        # reward_baseline: batch

        # Compute loss and accuracies
        additional_terms, selections = self.sample_candidate_terms(candidate_terms, probs, train=True)

        questions_extended = [question_str[i] + ' ' + ' '.join(doc_terms)
                              for i, doc_terms in enumerate(additional_terms)]
        results = self.search_engine.batch_closest_docs(questions_extended,
                                                        ranker_doc_max=self.args.ranker_doc_max,
                                                        candidate_doc_max=self.args.candidate_doc_max)
        docs_pred, _, _ = zip(*results)
        rewards_extended = [retrieve_metrics(doc_truth, doc_pred)
                            for (doc_truth, doc_pred) in zip(docs_truth, docs_pred)]
        rewards_extended_rec = [r['recall'] for r in rewards_extended]

        losses = []
        for i in range(len(questions)):
            # C_a
            cost_a = -(rewards_extended_rec[i] - rewards_recall[i] - reward_baseline[i]) * (probs[i].log() * selections[i]).sum()
            # C_b
            cost_b = self.args.stablize_alpha * (rewards_extended_rec[i] - rewards_recall[i] - reward_baseline[i]) ** 2
            # C_H
            cost_H = -self.args.entropy_regularizer * (probs[i] * probs[i].log()).sum()
            loss = cost_a + cost_b + cost_H
            losses.append(loss)
        losses = torch.cat(losses, dim=0).mean(0)

        # Clear gradients and run backward
        self.optimizer.zero_grad()

        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm(self.network.parameters(),
                                      self.args.grad_clipping)

        # Update parameters
        self.optimizer.step()
        self.updates += 1

        # Reset any partially fixed parameters (e.g. rare words)
        self.reset_parameters()

        return loss.data[0], len(ex[0]), rewards, rewards_extended

    # --------------------------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------------------------

    def retrieve(self, ex, top_n=1):
        """
        """
        # Eval mode
        self.network.eval()
        if len(ex) == 1:
            questions = ex
        else:
            questions, docs_truth = ex

        # vectorize questions
        questions_tensors = [torch.LongTensor([self.word_dict[w] for w in q]) for q in questions]
        # Batch questions
        max_q_length = max([q.size(0) for q in questions_tensors])
        x1 = torch.LongTensor(len(questions_tensors), max_q_length).zero_()
        x1_mask = torch.ByteTensor(len(questions_tensors), max_q_length).fill_(1)
        for i, q in enumerate(questions_tensors):
            x1[i, :q.size(0)].copy_(q)
            x1_mask[i, :q.size(0)].fill_(0)

        # retrieve docs
        question_str = [' '.join(q) for q in questions]
        results = self.search_engine.batch_closest_docs(
            question_str, ranker_doc_max=self.args.ranker_doc_max, candidate_doc_max=self.args.candidate_doc_max)

        docs_pred, term_idxs, terms = zip(*results)

        rewards = [retrieve_metrics(doc_truth, doc_pred)['recall']
                   for (doc_truth, doc_pred) in zip(docs_truth, docs_pred)]

        # remove empty doc terms
        term_idxs = term_idxs[:self.args.candidate_doc_max]
        terms = terms[:self.args.candidate_doc_max]

        # eval time: use all terms
        candidate_terms = [itertools.chain(*docs_term) for i, docs_term in enumerate(terms)]

        # pad head and tail with context window size
        pad_size = (self.args.context_window_size - 1) // 2
        term_idxs_pad = [[[0] * pad_size + d + [0] * pad_size for d in ds] for ds in term_idxs]
        term_idxs_ctxs = [[d[i - pad_size:i + pad_size + 1]
                           for d in ds for i in range(len(d)) if i >= pad_size and i <= len(d) - pad_size - 1]
                          for ds in term_idxs_pad]
        # term_idxs_ctxs: batch * terms * win_size
        max_terms_length = max([len(doc_terms) for doc_terms in term_idxs_ctxs])
        x2 = torch.LongTensor(len(term_idxs_ctxs), max_terms_length, self.args.context_window_size).zero_()
        x2_mask = torch.ByteTensor(len(term_idxs_ctxs), max_terms_length).fill_(1)

        for i, doc_terms in enumerate(term_idxs_ctxs):
            x2_mask[i, :len(doc_terms)].fill_(0)
            for j, term_contexts in enumerate(doc_terms):
                for k, word in enumerate(term_contexts):
                    x2[i, j, k] = term_idxs_ctxs[i][j][k]

        # Transfer to GPU
        if self.use_cuda:
            x1 = Variable(x1.cuda(async=True), volatile=True)
            x1_mask = Variable(x1_mask.cuda(async=True), volatile=True)
            x2 = Variable(x2.cuda(async=True), volatile=True)
            x2_mask = Variable(x2_mask.cuda(async=True), volatile=True)
        else:
            x1 = Variable(x1, volatile=True)
            x1_mask = Variable(x1_mask, volatile=True)
            x2 = Variable(x2, volatile=True)
            x2_mask = Variable(x2_mask, volatile=True)

        # Run forward
        probs, _ = self.network(x1, x1_mask, x2, x2_mask)
        # probs: batch * len

        # Compute loss and accuracies
        additional_terms, selections = self.sample_candidate_terms(
            candidate_terms, probs, train=False, epsilon=self.args.term_epsilon)
        questions_extended = [question_str[i] + ' ' + ' '.join(doc_terms)
                              for i, doc_terms in enumerate(additional_terms)]
        results = self.search_engine.batch_closest_docs(questions_extended,
                                                        ranker_doc_max=self.args.ranker_doc_max,
                                                        candidate_doc_max=self.args.candidate_doc_max)

        rewards = []
        if len(ex) == 1:
            return results, rewards
        else:
            docs_pred, _, _ = zip(*results)
            rewards_extended = [retrieve_metrics(doc_truth, doc_pred)
                                for (doc_truth, doc_pred) in zip(docs_truth, docs_pred)]
            return docs_pred, rewards_extended

    @staticmethod
    def sample_candidate_terms(docs, probs, train=True, epsilon=1):
        '''
        docs: [[str]]
        probs: Tensor(batch, len)

        return: [str], Tensor(batch, len)
        '''
        additional_terms = []
        if train:
            selections_var = torch.bernoulli(probs)
        else:
            selections_var = probs > epsilon
        selections = selections_var.data.cpu().numpy()
        for i, doc in enumerate(docs):
            doc_terms = []
            for j, word in enumerate(doc):
                if selections[i][j]:
                    doc_terms.append(word)
            additional_terms.append(doc_terms)

        return additional_terms, selections_var

    def reset_parameters(self):
        """Reset any partially fixed parameters to original states."""

        # Reset fixed embeddings to original value
        if self.args.tune_partial > 0:
            # Embeddings to fix are indexed after the special + N tuned words
            offset = self.args.tune_partial + self.word_dict.START
            if self.parallel:
                embedding = self.network.module.embedding.weight.data
                fixed_embedding = self.network.module.fixed_embedding
            else:
                embedding = self.network.embedding.weight.data
                fixed_embedding = self.network.fixed_embedding
            if offset < embedding.size(0):
                embedding[offset:] = fixed_embedding

    # --------------------------------------------------------------------------
    # Saving and loading
    # --------------------------------------------------------------------------

    def save(self, filename):
        state_dict = copy.copy(self.network.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {
            'state_dict': state_dict,
            'word_dict': self.word_dict,
            'args': self.args,
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    def checkpoint(self, filename, epoch):
        params = {
            'state_dict': self.network.state_dict(),
            'word_dict': self.word_dict,
            'args': self.args,
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    @staticmethod
    def load(filename, new_args=None, normalize=True):
        logger.info(f'Loading model {filename}')
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict']
        state_dict = saved_params['state_dict']
        args = saved_params['args']
        if new_args:
            args = override_model_args(args, new_args)
        return RLDocRetriever(args, word_dict, state_dict)

    @staticmethod
    def load_checkpoint(filename, normalize=True):
        logger.info(f'Loading model {filename}')
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict']
        state_dict = saved_params['state_dict']
        epoch = saved_params['epoch']
        optimizer = saved_params['optimizer']
        args = saved_params['args']
        model = RLDocRetriever(args, word_dict, state_dict)
        model.init_optimizer(optimizer)
        return model, epoch

    # --------------------------------------------------------------------------
    # Runtime
    # --------------------------------------------------------------------------

    def cuda(self):
        self.use_cuda = True
        self.network = self.network.cuda()

    def cpu(self):
        self.use_cuda = False
        self.network = self.network.cpu()

    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)
