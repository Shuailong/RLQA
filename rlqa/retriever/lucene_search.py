#!/usr/bin/env python
# encoding: utf-8
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
Use Lucene to retrieve candidate documents for given a query.
'''
import os
import logging
from multiprocessing.pool import ThreadPool

from tqdm import tqdm
from termcolor import colored

import lucene
from lucene import collections
from java.nio.file import Paths
from java.io import StringReader
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import DirectoryReader, IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.index import Term
from org.apache.lucene.store import MMapDirectory
from org.apache.lucene.store import FSDirectory, NIOFSDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.analysis import CharArraySet
from org.apache.lucene.search import PhraseQuery
from org.apache.lucene.search import BooleanQuery
from org.apache.lucene.search import BooleanClause
from org.apache.lucene.analysis.tokenattributes import CharTermAttribute
from org.apache.lucene.search.similarities import MyTFIDFSimilarity
from org.apache.lucene.analysis import MySimpleAnalyzer

from .. import tokenizers
from .. import DATA_DIR as RLQA_DATA
from . import utils
from .doc_db import DocDB


logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(RLQA_DATA, 'wikipedia')


class LuceneSearch(object):

    def __init__(self, args):

        self.env = lucene.initVM(initialheap='28g', maxheap='28g', vmargs=['-Djava.awt.headless=true'])
        self.args = args

        index_folder = os.path.join(DATA_DIR, args.index_folder)
        if not os.path.exists(index_folder):
            self.doc_db = DocDB()
            logger.info(f'Creating index at {index_folder}')
            self.create_index(index_folder)

        fsDir = MMapDirectory(Paths.get(index_folder))
        self.searcher = IndexSearcher(DirectoryReader.open(fsDir))
        self.searcher.setSimilarity(MyTFIDFSimilarity())
        self.analyzer = MySimpleAnalyzer(CharArraySet(collections.JavaSet(utils.STOPWORDS), True))
        self.pool = ThreadPool(processes=args.num_search_workers)


    def add_doc(self, title, text, tokens):

        doc = Document()
        doc.add(Field("title", title, self.t1))
        doc.add(Field("text", text, self.t2))
        doc.add(Field("token", tokens, self.t3))

        self.writer.addDocument(doc)

    def create_index(self, index_folder):
        os.mkdir(index_folder)

        self.t1 = FieldType()
        self.t1.setStored(True)
        self.t1.setIndexOptions(IndexOptions.DOCS)

        self.t2 = FieldType()
        self.t2.setStored(True)
        self.t2.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

        self.t3 = FieldType()
        self.t3.setStored(True)
        self.t3.setIndexOptions(IndexOptions.NONE)

        fsDir = MMapDirectory(Paths.get(index_folder))
        writerConfig = IndexWriterConfig(MySimpleAnalyzer(CharArraySet(collections.JavaSet(utils.STOPWORDS), True)))
        writerConfig.setSimilarity(MyTFIDFSimilarity())
        writerConfig.setRAMBufferSizeMB(16384.0)  # 14g
        self.writer = IndexWriter(fsDir, writerConfig)
        logger.info(f"{self.writer.numDocs()} docs in index")
        logger.info("Indexing documents...")

        doc_ids = self.doc_db.get_doc_ids()
        for doc_id in tqdm(doc_ids, total=len(doc_ids)):
            text = self.doc_db.get_doc_text(doc_id)
            tokens = self.doc_db.get_doc_tokens(doc_id)
            self.add_doc(doc_id, text, tokens)

        logger.info(f"Indexed {self.writer.numDocs()} docs.")
        self.writer.forceMerge(1)  # to increase search performance
        self.writer.close()

    def search_multithread(self, qs, ranker_doc_max, searcher):
        self.ranker_doc_max = ranker_doc_max
        self.curr_searcher = searcher
        out = self.pool.map(self.search_multithread_part, qs)

        return out

    def search_multithread_part(self, q):
        if not self.env.isCurrentThreadAttached():
            self.env.attachCurrentThread()

        try:
            if self.args.ngram == 2:
                query = self._parse_query(field_name='text', query=q)
            else:
                # self.args.ngram == 1
                query = QueryParser('text', self.analyzer).parse(QueryParser.escape(q))
        except Exception as e:
            logger.warning(colored(f'{e}: {q}, use query dummy.'), 'yellow')
            if self.args.ngram == 2:
                query = self._parse_query(field_name='text', query=q)
            else:
                # self.args.ngram == 1
                query = QueryParser('text', self.analyzer).parse('dummy')

        doc_scores, doc_titles, doc_texts, doc_words = [], [], [], []
        hits = self.curr_searcher.search(query, self.ranker_doc_max)

        for i, hit in enumerate(hits.scoreDocs):
            doc = self.curr_searcher.doc(hit.doc)

            doc_score = hit.score
            doc_title = doc['title']
            doc_word = doc['token'].split('<&>')
            doc_text = doc['text']

            doc_scores.append(doc_score)
            doc_titles.append(doc_title)
            doc_words.append(doc_word)
            doc_texts.append(doc_text)

        if len(doc_scores) == 0:
            logger.warning(colored(f'WARN: search engine returns no results for query: {q}.', 'yellow'))

        return doc_scores, doc_titles, doc_texts, doc_words

    def search_singlethread(self, qs, ranker_doc_max, curr_searcher):
        out = []
        for q in qs:
            try:
                if self.args.ngram == 2:
                    query = self._parse_query(field_name='text', query=q)
                else:
                    # self.args.ngram == 1
                    query = QueryParser('text', self.analyzer).parse(QueryParser.escape(q))
            except Exception as e:
                logger.warning(colored(f'{e}: {q}, use query dummy.'), 'yellow')
                if self.args.ngram == 2:
                    query = self._parse_query(field_name='text', query=q)
                else:
                    # self.args.ngram == 1
                    query = QueryParser('text', self.analyzer).parse('dummy')

            doc_scores, doc_titles, doc_texts, doc_words = [], [], [], []
            hits = curr_searcher.search(query, ranker_doc_max)

            for i, hit in enumerate(hits.scoreDocs):
                doc = curr_searcher.doc(hit.doc)

                doc_score = hit.score
                doc_title = doc['title']
                doc_word = doc['token'].split('<&>')
                doc_text = doc['text']

                doc_scores.append(doc_score)
                doc_titles.append(doc_title)
                doc_words.append(doc_word)
                doc_texts.append(doc_text)

            if len(doc_scores) == 0:
                logger.warning(colored(f'WARN: search engine returns no results for query: {q}.', 'yellow'))

            out.append((doc_scores, doc_titles, doc_texts, doc_words))

        return out

    def batch_closest_docs(self, qs, ranker_doc_max):

        if self.args.num_search_workers > 1:
            out = self.search_multithread(qs, ranker_doc_max, self.searcher)
        else:
            out = self.search_singlethread(qs, ranker_doc_max, self.searcher)

        return out

    def _parse_query(self, field_name, query):
        ts = self.analyzer.tokenStream("dummy", StringReader(query))
        termAtt = ts.getAttribute(CharTermAttribute.class_)
        ts.reset()
        tokens = []
        while ts.incrementToken():
            tokens.append(termAtt.toString())
        ts.end()
        ts.close()

        booleanQuery = BooleanQuery.Builder()
        for token in tokens:
            builder = PhraseQuery.Builder()
            for i, word in enumerate(token.split(' ')):
                builder.add(Term(field_name, word), i)
            pq = builder.build()
            booleanQuery.add(pq, BooleanClause.Occur.SHOULD)
        final_query = booleanQuery.build()
        return final_query

