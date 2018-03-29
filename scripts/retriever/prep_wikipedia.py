#!/usr/bin/env python
# encoding: utf-8
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# Adapt from facebookresearch/DrQA by Shuailong on Mar 22 2018.

"""Preprocess function to filter/prepare Wikipedia docs."""

import regex as re
from html.parser import HTMLParser

PARSER = HTMLParser()
BLACKLIST = set(['23443579', '52643645'])  # Conflicting disambig. pages


def preprocess(article):
    # Take out HTML escaping WikiExtractor didn't clean
    for k, v in article.items():
        article[k] = PARSER.unescape(v)

    # Filter some disambiguation pages not caught by the WikiExtractor
    if article['id'] in BLACKLIST:
        return None
    if '(disambiguation)' in article['title'].lower():
        return None
    if '(disambiguation page)' in article['title'].lower():
        return None

    # Take out List/Index/Outline pages (mostly links)
    if re.match(r'(List of .+)|(Index of .+)|(Outline of .+)',
                article['title']):
        return None

    # Return doc with `id` set to `title`
    return {'id': article['title'], 'text': article['text']}
