#!/usr/bin/env python
# encoding: utf-8
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Functions for putting examples into torch format."""

from collections import Counter
import torch


def batchify(batch):
    """Gather a batch of individual examples into one batch."""

    questions = [ex['question'] for ex in batch]
    docs_truth = [ex['doc_truth'] for ex in batch]
    return questions, docs_truth

