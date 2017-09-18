#!/usr/bin/env python
# -*- coding:utf-8 -*-

from pprint import pprint
from collections import Counter
from janome.tokenizer import Tokenizer
from math import log

tokeniser = Tokenizer()

def print_detail(text):
    for token in tokeniser.tokenize(text):
        print()
        print(token)
        props = {
            'base_form': token.base_form,
            'infl_form': token.infl_form,
            'infl_type': token.infl_type,
            'node_type': token.node_type,
            'part_of_speech': token.part_of_speech,
            'phonetic': token.phonetic,
            'reading': token.reading,
            'surface': token.surface,
        }
        pprint(props)

def morphological_analysis(text):
    token_list = []
    for token in tokeniser.tokenize(text):
        token_list.append(token.surface)
    return token_list

def unigram(text):
    return list(text)

def bigram(text):
    return [ a + b for a, b in zip(text[:-1], text[1:])]

def bag_of_words(token_list):
    cnt = Counter()
    for token in token_list:
        cnt[token] += 1

    for k, v in cnt.most_common():
        print(k, v)

def tf_idf(document_list, make_token_func=unigram):
    df_cnt = Counter()
    tf_cnt_list = []
    for document in document_list:
        tf_cnt = Counter()
        token_set = set()
        for token in make_token_func(document):
            tf_cnt[token] += 1
            token_set.add(token)

        for token in token_set:
            df_cnt[token] += 1
        tf_cnt_list.append(tf_cnt)
        

    
    print('tf-idf score')
    num_document = len(document_list)
    for tf_cnt, document in zip(tf_cnt_list, document_list):
        print(document)
        for k, v in tf_cnt.most_common():
            print(k,
                'tf:', v,
                'idf:', log(num_document) - log(df_cnt[k]),
                'tf-idf:', v * (log(num_document) - log(df_cnt[k]))
            )

if __name__ == '__main__':
    text = '吾輩は猫である'
    result = morphological_analysis(text)
    print(result)
    result = bigram(text)
    print(result)
    result = unigram(text)
    print(result)
    document_list = [
        'すもももももももものうち',
        'おどろき、もものき、さんしょのき',
        'ももくりさんねんかきはちねん',
    ]
    bag_of_words(document_list)
    tf_idf(document_list)
