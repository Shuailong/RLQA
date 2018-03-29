#!/usr/bin/env python
# encoding: utf-8
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Implementation of the RNN based RLQA reader."""


import torch.nn as nn
from . import layers


class Reformulator(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, args):
        super(Reformulator, self).__init__()
        # Store config
        self.args = args

        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)
        # RNN question encoder
        self.question_rnn = layers.StackedBRNN(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.question_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        # RNN document encoder
        self.doc_rnn = layers.StackedBRNN(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.doc_layers,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        self.value_net = layers.MLP(args)
        self.actor_net = layers.MLP(args)

    def forward(self, x1, x1_mask, x2, x2_mask):
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)

        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.args.dropout_emb,
                                           training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.args.dropout_emb,
                                           training=self.training)

        # Encode question with RNN + merge hiddens
        question_hiddens = self.question_rnn(x1_emb, x1_mask)
        # question_hiddens: batch * len * hdim
        weights = layers.uniform_weights(question_hiddens, x1_mask)
        question_repr = layers.weighted_avg(question_hiddens, weights)
        # question_repr: batch * hdim

        # Encode document with RNN
        doc_hiddens = self.doc_rnn(x2_emb, x2_mask)
        # doc_hiddens: batch * len * hdim
        weights = layers.uniform_weights(doc_hiddens, x2_mask)
        doc_avg_repr = layers.weighted_avg(doc_hiddens, weights)
        # doc_avg_repr: batch * hdim

        reward_baseline = self.value_net(question_repr, doc_avg_repr)
        question_expand = question_repr.unsqueeze(1).expand_as(doc_hiddens)
        # question_expand: batch * len * hdim

        probs = self.actor_net(question_expand, doc_hiddens)
        # probs: batch * len

        return probs, reward_baseline
