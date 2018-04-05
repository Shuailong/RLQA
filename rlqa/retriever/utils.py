#!/usr/bin/env python
# encoding: utf-8
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Various retriever utilities."""

import json
import time
import logging
import unicodedata
import regex as re
from collections import Counter

import numpy as np
import scipy.sparse as sp
from sklearn.utils import murmurhash3_32

from .data import Dictionary


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Sparse matrix saving/loading helpers.
# ------------------------------------------------------------------------------


def save_sparse_csr(filename, matrix, metadata=None):
    data = {
        'data': matrix.data,
        'indices': matrix.indices,
        'indptr': matrix.indptr,
        'shape': matrix.shape,
        'metadata': metadata,
    }
    np.savez(filename, **data)


def load_sparse_csr(filename):
    loader = np.load(filename)
    matrix = sp.csr_matrix((loader['data'], loader['indices'],
                            loader['indptr']), shape=loader['shape'])
    return matrix, loader['metadata'].item(0) if 'metadata' in loader else None


# ------------------------------------------------------------------------------
# Token hashing.
# ------------------------------------------------------------------------------

def hash(token, num_buckets):
    """Unsigned 32 bit murmurhash for feature hashing."""
    return murmurhash3_32(token, positive=True) % num_buckets


# ------------------------------------------------------------------------------
# Text cleaning.
# ------------------------------------------------------------------------------


STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've',
    'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
    'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren',
    'won', 'wouldn', "'ll", "'re", "'ve", "n't", "'s", "'d", "'m", "''", "``"
}


def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


def clean(txt):
    '''
    # remove most of Wikipedia and AQUAINT markups, such as '[[', and ']]'.
    '''
    txt = re.sub(r'\|.*?\]\]', '', txt)  # remove link anchor
    txt = txt.replace('&amp;', ' ').replace('&lt;', ' ').replace('&gt;', ' ')\
        .replace('&quot;', ' ').replace('\'', ' ').replace('(', ' ')\
        .replace(')', ' ').replace('.', ' ').replace('"', ' ')\
        .replace(',', ' ').replace(';', ' ').replace(':', ' ')\
        .replace('<93>', ' ').replace('<98>', ' ').replace('<99>', ' ')\
        .replace('<9f>', ' ').replace('<80>', ' ').replace('<82>', ' ')\
        .replace('<83>', ' ').replace('<84>', ' ').replace('<85>', ' ')\
        .replace('<89>', ' ').replace('=', ' ').replace('*', ' ')\
        .replace('\n', ' ').replace('!', ' ').replace('-', ' ')\
        .replace('[[', ' ').replace(']]', ' ')

    return txt


def filter_word(text):
    """Take out english stopwords, punctuation, and compound endings."""
    text = normalize(text)
    if re.match(r'^\p{P}+$', text):
        return True
    if text.lower() in STOPWORDS:
        return True
    return False


def filter_ngram(gram, mode='any'):
    """Decide whether to keep or discard an n-gram.

    Args:
        gram: list of tokens (length N)
        mode: Option to throw out ngram if
          'any': any single token passes filter_word
          'all': all tokens pass filter_word
          'ends': book-ended by filterable tokens
    """
    filtered = [filter_word(w) for w in gram]
    if mode == 'any':
        return any(filtered)
    elif mode == 'all':
        return all(filtered)
    elif mode == 'ends':
        return filtered[0] or filtered[-1]
    else:
        raise ValueError('Invalid mode: %s' % mode)


# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------


def load_data(filename):
    """Load examples from preprocessed file.
    One example per line, JSON encoded.
    """
    # Load JSON lines
    with open(filename, encoding='utf-8') as f:
        examples = [json.loads(line) for line in f]

    return examples


# ------------------------------------------------------------------------------
# Dictionary building
# ------------------------------------------------------------------------------

def index_embedding_words(embedding_file):
    """Put all the words in embedding_file into a set."""
    words = set()
    with open(embedding_file, encoding='utf-8') as f:
        for line in f:
            w = Dictionary.normalize(line.rstrip().split(' ')[0])
            words.add(w)
    return words


def load_words(args, examples):
    """Iterate and index all the words in examples (questions)."""
    if args.restrict_vocab and args.embedding_file:
        logger.info(f'Restricting to words in {args.embedding_file}')
        valid_words = index_embedding_words(args.embedding_file)
        logger.info(f'Num words in set = {len(valid_words)}')
    else:
        valid_words = None

    words = set()
    for ex in examples:
        for word in ex['question']:
            word = Dictionary.normalize(word)
            if valid_words and word not in valid_words:
                continue
            if args.uncased_question:
                word = word.lower()
            words.add(word)
        for ans in ex['answer']:
            for word in Dictionary.normalize(word):
                if valid_words and word not in valid_words:
                    continue
                if args.uncased_doc:
                    word = word.lower()
                words.add(word)
    return words


def build_word_dict(args, examples):
    """Return a dictionary from question and document words in
    provided examples.
    """
    word_dict = Dictionary()
    for w in load_words(args, examples):
        word_dict.add(w)
    return word_dict


def top_question_words(args, examples, word_dict):
    """Count and return the most common question words in provided examples."""
    word_count = Counter()
    for ex in examples:
        for w in ex['question']:
            w = Dictionary.normalize(w)
            if args.uncased_question:
                w = w.lower()
            if w in word_dict:
                word_count.update([w])
    return word_count.most_common(args.tune_partial)


# ------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------

def average_precision(title_truth, title_pred):
    '''Computes the average precision.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision over the input lists
    '''
    if not title_truth:
        return 0.0

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(title_pred):
        if p in title_truth and p not in title_pred[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / max(1, min(len(title_pred), len(title_truth)))


def metrics_by_title(title_truth, title_pred):
    """Search through all the top docs to see if they have the answer."""
    TP = len(set(title_truth) & set(title_pred))
    precision = TP / len(title_pred)
    recall = TP / len(title_truth)
    F1 = 2 * precision * recall / max(0.01, recall + precision)
    hit = 1 if precision > 0 else 0
    MAP = average_precision(title_truth, title_pred)
    metrics = {'precision': precision, 'recall': recall, 'F1': F1, 'map': MAP, 'hit': hit}
    return metrics


def metrics_by_content(answer, doc_pred, match='string'):
    """Search through all the top docs to see if they have the answer."""

    def regex_match(text, pattern):
        """Test if a regex pattern is contained within a text."""
        try:
            pattern = re.compile(
                pattern,
                flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
            )
        except BaseException:
            return False
        return pattern.search(text) is not None

    def has_answer(answer, doc, match):
        if match == 'string':
            for single_answer in answer:
                for i in range(len(doc) - len(single_answer) - 1):
                    if single_answer == doc[i: i + len(single_answer)]:
                        return True
        elif match == 'regex':
            single_answer = answer[0]
            if regex_match(doc, single_answer):
                return True
        return False

    TP = 0
    for doc in doc_pred:
        if has_answer(answer, doc, match):
            TP += 1

    hit = 1 if TP > 0 else 0
    precision = TP / len(doc_pred)

    recall = F1 = MAP = -1

    metrics = {'precision': precision, 'recall': recall, 'F1': F1, 'map': MAP, 'hit': hit}
    return metrics


# ------------------------------------------------------------------------------
# Utility classes
# ------------------------------------------------------------------------------

class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total
