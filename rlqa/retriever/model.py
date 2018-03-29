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
import regex as re

import numpy as np

import torch
from torch.autograd import Variable
import torch.optim as optim

from .config import override_model_args
from .reformulator import Reformulator
from .tfidf_doc_ranker import TfidfDocRanker
from .doc_db import DocDB
from . import utils
from .. import tokenizers

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

        self.doc_ranker = TfidfDocRanker()
        self.db = DocDB()
        self.tok = tokenizers.get_class(args.tokenizer)()
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

        questions, answers = ex
        # retrieve docs
        results = self.doc_ranker.batch_closest_docs(questions)
        # reward = utils.retrieve_metrics(retrieved, answer)['precision']
        candidate_articles = self.sample_candidate_article(results)
        articles_text = [self.db.get_doc_text(doc_id) for (doc_id, _) in candidate_articles]
        # vectorize questions and docs
        question_tokens = [self.tok.tokenize(q).words() for q in questions]
        if self.args.uncased_question:
            question_tokens = [[w.lower() for w in q]for q in questions]
        questions_tensors = [torch.LongTensor([self.word_dict[w] for w in q]) for q in question_tokens]

        doc_tokens = [self.tok.tokenize(doc).words() for doc in articles_text]
        if self.args.uncased_doc:
            doc_tokens = [[w.lower() for w in d]for d in doc_tokens]
        docs = [torch.LongTensor([self.word_dict[w] for w in doc]) for doc in doc_tokens]

        # Batch questions
        max_length = max([q.size(0) for q in questions_tensors])
        x1 = torch.LongTensor(len(questions_tensors), max_length).zero_()
        x1_mask = torch.ByteTensor(len(questions_tensors), max_length).fill_(1)
        for i, q in enumerate(questions_tensors):
            x1[i, :q.size(0)].copy_(q)
            x1_mask[i, :q.size(0)].fill_(0)

        # Batch documents
        max_length = max([d.size(0) for d in docs])
        x2 = torch.LongTensor(len(docs), max_length).zero_()
        x2_mask = torch.ByteTensor(len(docs), max_length).fill_(1)

        for i, d in enumerate(docs):
            x2[i, :d.size(0)].copy_(d)
            x2_mask[i, :d.size(0)].fill_(0)

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
        additional_terms, selections = self.sample_candidate_terms(doc_tokens, probs, train=True)

        questions_extended = [questions[i] + ' ' + ' '.join(doc_terms) for i, doc_terms in enumerate(additional_terms)]
        results = self.doc_ranker.batch_closest_docs(questions_extended)

        answer_docs = zip(answers, results)
        rewards_extended = []
        for answer_doc in answer_docs:
            rewards_extended.append(self.get_score(answer_doc)['precision'])

        losses = []
        for i in range(len(answers)):
            # C_a
            cost_a = -(rewards_extended[i] - reward_baseline[i]) * (probs.log() * selections).sum(1)[i]
            # C_b
            cost_b = self.args.stablize_alpha * (rewards_extended[i] - reward_baseline[i]) ** 2
            # C_H
            cost_H = -self.args.entropy_regularizer * (probs * probs.log()).sum(1)[i]
            loss = cost_a + cost_b + cost_H
            losses.append(loss)
        losses = torch.cat(losses, dim=-1).mean(0)

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

        return loss.data[0], len(ex[0])

    # --------------------------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------------------------

    def retrieve(self, ex, top_n=1):
        """Forward a batch of examples only to get predictions.

        Args:
            ex: the batch
            top_n: Number of predictions to return per batch element.
            async_pool: If provided, non-gpu post-processing will be offloaded
              to this CPU process pool.
        Output:

        If async_pool is given, these will be AsyncResult handles.
        """
        # Eval mode
        self.network.eval()
        if len(ex) == 1:
            questions = ex
        else:
            questions, answers = ex

        # retrieve docss
        results = self.doc_ranker.batch_closest_docs(questions)
        candidate_articles = self.sample_candidate_article(results)
        articles_text = [self.db.get_doc_text(doc_id) for (doc_id, _) in candidate_articles]
        # vectorize questions and docs
        question_tokens = [self.tok.tokenize(q).words() for q in questions]
        if self.args.uncased_question:
            question_tokens = [[w.lower() for w in q]for q in questions]
        questions = [torch.LongTensor([self.word_dict[w] for w in q]) for q in question_tokens]

        doc_tokens = [self.tok.tokenize(doc).words() for doc in articles_text]
        if self.args.uncased_doc:
            doc_tokens = [[w.lower() for w in d]for d in doc_tokens]
        docs = [torch.LongTensor([self.word_dict[w] for w in doc]) for doc in doc_tokens]

        # Batch questions
        max_length = max([q.size(0) for q in questions])
        x1 = torch.LongTensor(len(questions), max_length).zero_()
        x1_mask = torch.ByteTensor(len(questions), max_length).fill_(1)
        for i, q in enumerate(questions):
            x1[i, :q.size(0)].copy_(q)
            x1_mask[i, :q.size(0)].fill_(0)

        # Batch documents
        max_length = max([d.size(0) for d in docs])
        x2 = torch.LongTensor(len(docs), max_length).zero_()
        x2_mask = torch.ByteTensor(len(docs), max_length).fill_(1)

        for i, d in enumerate(docs):
            x2[i, :d.size(0)].copy_(d)
            x2_mask[i, :d.size(0)].fill_(0)

        # Transfer to GPU
        if self.use_cuda:
            x1 = Variable(x1.cuda(async=True))
            x2 = Variable(x2.cuda(async=True))
        else:
            x1 = Variable(x1)
            x2 = Variable(x2)

        # Run forward
        probs, _ = self.network(x1, x1_mask, x2, x2_mask)
        # probs: batch * len

        # Compute loss and accuracies
        additional_terms, selections = self.sample_candidate_terms(
            doc_tokens, probs, train=False, epsilon=self.args.term_epsilon)
        questions_extended = [questions[i] + ' ' + ' '.join(doc_terms) for i, doc_terms in enumerate(additional_terms)]
        results = self.doc_ranker.batch_closest_docs(questions_extended)

        rewards = []
        if len(ex) == 1:
            return results, rewards
        else:
            answer_docs = zip(answers, results)
            for answer_doc in answer_docs:
                rewards.append(self.get_score(answer_doc))
            return results, rewards

    def regex_match(self, text, pattern):
        """Test if a regex pattern is contained within a text."""
        try:
            pattern = re.compile(
                pattern,
                flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
            )
        except BaseException:
            return False
        return pattern.search(text) is not None

    def has_answer(self, answer, doc_id):
        """Check if a document contains an answer string.

        If `match` is string, token matching is done between the text and answer.
        If `match` is regex, we search the whole text with the regex.
        """
        text = self.db.get_doc_text(doc_id)
        text = utils.normalize(text)
        if self.args.match == 'string':
            # Answer is a list of possible strings
            text = self.tok.tokenize(text).words(uncased=True)
            if isinstance(answer, str):
                answer = [answer]
            for single_answer in answer:
                single_answer = utils.normalize(single_answer)
                single_answer = self.tok.tokenize(single_answer)
                single_answer = single_answer.words(uncased=True)
                for i in range(0, len(text) - len(single_answer) + 1):
                    if single_answer == text[i: i + len(single_answer)]:
                        return True
        elif self.args.match == 'regex':
            # Answer is a regex
            single_answer = utils.normalize(answer[0])
            if self.regex_match(text, single_answer):
                return True
        return False

    def get_score(self, answer_doc):
        """Search through all the top docs to see if they have the answer."""
        answer, (doc_ids, doc_scores) = answer_doc
        N = len(doc_ids)
        precision, hit, MAP = None, None, None
        TP = 0
        for doc_id in doc_ids:
            if self.has_answer(answer, doc_id):
                TP += 1
        precision = TP / N
        if TP > 0:
            hit = 1
        metrics = {'precision': precision, 'map': MAP, 'hit': hit}
        return metrics

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

    @staticmethod
    def sample_candidate_article(results):
        '''
        results: output of TfidfDocRanker.batch_closest_docs
                 [(doc_ids, doc_scores)]
        '''
        samples = [(doc_ids[0], doc_scores[0]) for (doc_ids, doc_scores) in results]
        return samples

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
