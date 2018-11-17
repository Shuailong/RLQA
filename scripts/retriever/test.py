# !/usr/bin/env python
# encoding: utf-8
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Main RLQA retriever code test script."""

# ------------------------------------------------------------------------------
# data.py.
# ------------------------------------------------------------------------------

# from rlqa.retriever.data import Dictionary
# from rlqa.retriever.utils import normalize
# from rlqa.tokenizers import CoreNLPTokenizer

# tok = CoreNLPTokenizer()
# d = Dictionary()
# sent = '''
# A general study of the effect of depth on optimization entails an inherent difficulty - deeper networks may seem to converge faster due to their superior expressiveness. In other words, if optimization of a deep network progresses more rapidly than that of a shallow one, it may not be obvious whether this is a result of a true acceleration phenomenon, or simply a byproduct of the fact that the shallow model cannot reach the same loss as the deep one. We resolve this conundrum by focusing on models whose representational capacity is oblivious to depth - linear neural networks, the subject of many recent studies. With linear networks, adding layers does not alter expressiveness; it manifests itself only in the replacement of a matrix parameter by a product of matrices - an overparameterization. Accordingly, if this leads to accelerated convergence, one can be certain that it is not an outcome of any phenomenon other than favorable properties of depth for optimization.
# '''
# words = tok.tokenize(sent).words(uncased=True)
# for word in words:
#     d.add(normalize(word))
# print(sent)
# print('vocab:', len(d))
# print('tokens:', d.tokens())

# ------------------------------------------------------------------------------
# doc_db.py.
# ------------------------------------------------------------------------------

# from time import time
# from rlqa.retriever.doc_db import DocDB
# from random import sample
# from tqdm import tqdm

# start = time()
# db = DocDB()
# doc_ids = sample(db.get_doc_ids(), 100)
# for doc_id in tqdm(doc_ids, total=len(doc_ids)):
#     text = db.get_doc_text(doc_id)
# end = time()
# print(f'{(end - start)/100:.2f} seconds/sample.')
# 0.04s/ sample

# ------------------------------------------------------------------------------
# layers.py.
# ------------------------------------------------------------------------------

# from rlqa.retriever.layers import StackedBRNN, MLP
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# import numpy as np

# batch_size = 10
# vocab_size = 20
# max_len = 200
# emb_size = 300
# hid_size = 256
# layers = 2

# rnn = StackedBRNN(emb_size, hid_size, layers,
#                   dropout_rate=0,
#                   dropout_output=False,
#                   rnn_type=nn.LSTM,
#                   concat_layers=False,
#                   padding=False)
# emb = nn.Embedding(vocab_size, emb_size, padding_idx=0)

# data = np.random.randint(vocab_size, size=(batch_size, max_len))
# mask = Variable(torch.zeros(1))
# v = Variable(torch.from_numpy(data))
# print(f'v size: {v.size()}')
# emb_v = emb(v)
# print(f'emb size: {emb_v.size()}')
# hid_v = rnn(emb_v, mask)
# print(f'hid size: {hid_v.size()}')
# rep = torch.cat([hid_v[:, -1, :hid_size], hid_v[:, 0, -hid_size:]], dim=-1)
# print(f'rep size: {rep.size()}')

# srnn = nn.LSTM(emb_size, hid_size, layers, bidirectional=True, batch_first=True)
# hid_vs, _ = srnn(emb_v)
# print(f'shid size: {hid_vs.size()}')
# rep = torch.cat([hid_v[:, -1, :hid_size], hid_v[:, 0, -hid_size:]], dim=-1)
# print(f'rep size: {rep.size()}')

# ------------------------------------------------------------------------------
# utils.py.
# ------------------------------------------------------------------------------

# from rlqa.retriever import utils
# answer = [['Goldman', 'Sachs'], ['Bank', 'of', 'US']]
# doc_pred = [['Goldman', 'Sachs', 'haha'], 
#             ['A', 'Bank', 'of', 'US'],
#             ['Goldman', 'Sachs', 'aa'],
#             ['aa', 'bb']]
# match = 'string'
# metrics = utils.metrics_by_content(answer, doc_pred, match=match)
# print(metrics)

# ------------------------------------------------------------------------------
# utils.py.
# ------------------------------------------------------------------------------

from rlqa.retriever import DocDB

db = DocDB(db_path='data/wikipedia/docs_tokens.db')
ids = db.get_doc_ids()[:3]
for i in ids:
    text = db.get_doc_text(i)
    tokens = db.get_doc_tokens(i).split('<&>')
    print('---')


