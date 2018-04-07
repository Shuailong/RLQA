#!/usr/bin/env python
# encoding: utf-8
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# Adapt from facebookresearch/DrQA by Shuailong on Mar 22 2018.

"""A script to read in and store documents in a sqlite database."""

import argparse
import sqlite3
import os
import logging

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from tqdm import tqdm

from rlqa import tokenizers, retriever
from rlqa import DATA_DIR


WIKI_DIR = os.path.join(DATA_DIR, 'wikipedia')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


# ------------------------------------------------------------------------------
# Multiprocessing target functions.
# ------------------------------------------------------------------------------

PROCESS_TOK = None
PROCESS_DB = None


def init(tokenizer_class, tokenizer_opts, db_class, db_opts):
    global PROCESS_TOK, PROCESS_DB
    PROCESS_TOK = tokenizer_class(**tokenizer_opts)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)

# ------------------------------------------------------------------------------
# Retrive content and tokenize.
# ------------------------------------------------------------------------------

def tokenize_content(doc_id):
    """ """
    global PROCESS_DB, PROCESS_TOK
    doc_text = PROCESS_DB.get_doc_text(doc_id)
    doc_tokens = PROCESS_TOK.tokenize(retriever.utils.normalize(doc_text)).words(uncased=True)
    return doc_id, doc_text, '<&>'.join(doc_tokens)

def get_doc_ids():
    global PROCESS_DB
    return PROCESS_DB.get_doc_ids()


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-file', type=str, default='docs_tokens.db')
    parser.add_argument('--doc-db', type=str, default=None,
                        help='Path to Document DB')
    parser.add_argument('--tokenizer', type=str, default='regexp')
    parser.add_argument('--num-workers', type=int, default=32,
                        help='Number of CPU processes (for tokenizing, etc)')
    args = parser.parse_args()

    save_path = os.path.join(WIKI_DIR, args.save_file)
    logger.info(f'Database will be saved in {save_path}')

    # define processes
    tok_class = tokenizers.get_class(args.tokenizer)
    tok_opts = {}
    db_class = retriever.DocDB
    db_opts = {'db_path': args.doc_db}
    processes = ProcessPool(
        processes=args.num_workers,
        initializer=init,
        initargs=(tok_class, tok_opts, db_class, db_opts)
    )

    logger.info('Reading into database...')
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    c.execute("CREATE TABLE documents (id PRIMARY KEY, text, tokens);")

    doc_ids = processes.apply(get_doc_ids)

    count = 0
    with tqdm(total=len(doc_ids)) as pbar:
        for pairs in tqdm(processes.imap_unordered(tokenize_content, doc_ids)):
            count += len(pairs)
            c.execute("INSERT INTO documents VALUES (?,?,?)", pairs)
            pbar.update()
    logger.info('Read %d docs.' % count)
    logger.info('Committing...')
    conn.commit()
    conn.close()
