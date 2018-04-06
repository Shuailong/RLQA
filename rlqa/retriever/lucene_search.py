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
import itertools
import pickle
from multiprocessing.pool import ThreadPool

from tqdm import tqdm

import lucene
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import DirectoryReader, IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.store import MMapDirectory
from org.apache.lucene.search import IndexSearcher, MatchAllDocsQuery
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search.similarities import BM25Similarity, ClassicSimilarity

from .. import tokenizers
from .. import DATA_DIR as RLQA_DATA
from . import utils
from .doc_db import DocDB


logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(RLQA_DATA, 'wikipedia')


class LuceneSearch(object):

    def __init__(self, args):

        self.env = lucene.initVM()
        self.args = args
        self.tokenizer = tokenizers.get_class(args.tokenizer)()
        self.doc_db = DocDB()

        index_folder = os.path.join(DATA_DIR, args.index_folder)
        if not os.path.exists(index_folder):
            logger.info(f'Creating index at {index_folder}')
            self.create_index(index_folder, add_terms=True)

        fsDir = MMapDirectory(Paths.get(index_folder))
        self.searcher = IndexSearcher(DirectoryReader.open(fsDir))
        if args.similarity == 'classic':
            logger.info('Use Classic similarity for lucene searcher.')
            self.searcher.setSimilarity(ClassicSimilarity())
        else:
            logger.info('Use BM25 similarity for lucene searcher.')
            self.searcher.setSimilarity(BM25Similarity())
        self.analyzer = StandardAnalyzer()
        self.pool = ThreadPool(processes=args.num_search_workers)
        self.cache = {}

        logger.info('Loading Title-ID mapping...')
        cache_file = os.path.join(DATA_DIR, 'title-id.pkl')
        if os.path.isfile(cache_file):
            logger.info(f'Cache file {cache_file} exists. Load from cache file.')
            self.title_id_map, self.id_title_map = pickle.load(open(cache_file, 'rb'))
        else:
            self.title_id_map, self.id_title_map = self.get_title_id_map()
            pickle.dump((self.title_id_map, self.id_title_map), open(cache_file, 'wb'))
            logger.info(f'Dump Title-ID mapping into {cache_file}.')

    def get_title_id_map(self):

        # get number of docs
        n_docs = self.searcher.getIndexReader().numDocs()

        title_id = {}
        id_title = {}
        query = MatchAllDocsQuery()
        hits = self.searcher.search(query, n_docs)
        for hit in tqdm(hits.scoreDocs, total=n_docs):
            doc = self.searcher.doc(hit.doc)
            idd = int(doc['id'])
            title = doc['title']
            title_id[title] = idd
            id_title[idd] = title

        return title_id, id_title

    def add_doc(self, doc_idx, title, txt, add_terms):

        doc = Document()
        doc.add(Field("id", str(doc_idx), self.t1))
        doc.add(Field("title", title, self.t1))
        doc.add(Field("text", txt, self.t2))

        if add_terms:
            words = self.tokenizer.tokenize(utils.normalize(txt)).words(uncased=True)
            doc.add(Field("word", '<&>'.join(words), self.t3))
            doc.add(Field("fulltext", txt, self.t3))

        self.writer.addDocument(doc)

    def create_index(self, index_folder, add_terms=False):
        os.mkdir(index_folder)

        self.t1 = FieldType()
        self.t1.setStored(True)
        self.t1.setIndexOptions(IndexOptions.DOCS)

        self.t2 = FieldType()
        self.t2.setStored(False)
        self.t2.setIndexOptions(IndexOptions.DOCS_AND_FREQS)

        self.t3 = FieldType()
        self.t3.setStored(True)
        self.t3.setIndexOptions(IndexOptions.NONE)

        fsDir = MMapDirectory(Paths.get(index_folder))
        writerConfig = IndexWriterConfig(StandardAnalyzer())
        if self.args.similarity == 'classic':
            logger.info('Use Classic similarity for lucene writer.')
            writerConfig.setSimilarity(ClassicSimilarity())
        else:
            logger.info('Use BM25 similarity for lucene writer.')
            writerConfig.setSimilarity(BM25Similarity())
        self.writer = IndexWriter(fsDir, writerConfig)
        logger.info(f"{self.writer.numDocs()} docs in index")
        logger.info("Indexing documents...")

        doc_ids = self.doc_db.get_doc_ids()
        for idx, doc_id in tqdm(enumerate(doc_ids), total=len(doc_ids)):
            txt = self.doc_db.get_doc_text(doc_id)
            self.add_doc(idx, doc_id, txt, add_terms)

        logger.info(f"Index of {self.writer.numDocs()} docs...")
        self.writer.close()

    def search_multithread(self, qs, ranker_doc_max, searcher):
        self.ranker_doc_max = ranker_doc_max
        self.curr_searcher = searcher
        out = self.pool.map(self.search_multithread_part, qs)

        return out

    def search_multithread_part(self, q):
        if not self.env.isCurrentThreadAttached():
            self.env.attachCurrentThread()

        if q in self.cache:
            return self.cache[q]
        else:
            try:
                q = q.replace('AND', '\\AND').replace('OR', '\\OR').replace('NOT', '\\NOT')
                query = QueryParser("text", self.analyzer).parse(QueryParser.escape(q))
            except Exception:
                logger.info(f'Unexpected error when processing query: {str(q)}')
                logger.info('Using query "dummy".')
                q = 'dummy'
                query = QueryParser("text", self.analyzer).parse(QueryParser.escape(q))

            doc_scores, doc_titles, doc_texts, doc_words = [], [], [], []
            hits = self.curr_searcher.search(query, self.ranker_doc_max)

            for i, hit in enumerate(hits.scoreDocs):
                doc = self.curr_searcher.doc(hit.doc)

                doc_score = hit.score
                doc_title = self.id_title_map[int(doc['id'])]
                doc_word = doc['word'].split('<&>')
                doc_text = doc['fulltext']

                doc_scores.append(doc_score)
                doc_titles.append(doc_title)
                doc_words.append(doc_word)
                doc_texts.append(doc_text)
            return doc_scores, doc_titles, doc_texts, doc_words

    def search_singlethread(self, qs, ranker_doc_max, curr_searcher):
        out = []
        for q in qs:
            if q in self.cache:
                out.append(self.cache[q])
            else:
                try:
                    q = q.replace('AND', '\\AND').replace('OR', '\\OR').replace('NOT', '\\NOT')
                    query = QueryParser("text", self.analyzer).parse(QueryParser.escape(q))
                except Exception:
                    logger.info(f'Unexpected error when processing query: {str(q)}')
                    logger.info('Using query "dummy".')
                    query = QueryParser("text", self.analyzer).parse(QueryParser.escape('dummy'))

                doc_scores, doc_titles, doc_texts, doc_words = [], [], [], []
                hits = curr_searcher.search(query, ranker_doc_max)

                for i, hit in enumerate(hits.scoreDocs):
                    doc = self.curr_searcher.doc(hit.doc)

                    doc_score = hit.score
                    doc_title = self.id_title_map[int(doc['id'])]
                    doc_word = doc['word'].split('<&>')
                    doc_text = doc['fulltext']

                    doc_scores.append(doc_score)
                    doc_titles.append(doc_title)
                    doc_words.append(doc_word)
                    doc_texts.append(doc_text)

                out.append((doc_scores, doc_titles, doc_texts, doc_words))

        return out

    def batch_closest_docs(self, qs, ranker_doc_max, save_cache=False):

        if self.args.num_search_workers > 1:
            out = self.search_multithread(qs, ranker_doc_max, self.searcher)
        else:
            out = self.search_singlethread(qs, ranker_doc_max, self.searcher)

        if save_cache:
            for q, c in itertools.izip(qs, out):
                if q not in self.cache:
                    self.cache[q] = c

        return out
