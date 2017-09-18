#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
from six.moves import urllib
from gensim.models.word2vec import Word2Vec
import numpy as np
from natsume_loader import token_generator, read_data

url = 'http://www.aozora.gr.jp/cards/000148/files/'


def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urllib.request.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception(
        'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

# 夏目漱石の小説データのダウンロード
maybe_download('789_ruby_5639.zip', 350404) # 吾輩は猫である
maybe_download('752_ruby_2438.zip', 94410) # 坊っちゃん

# ダウンロードしたデータを形態素解析
wagahaiwa = list(token_generator(read_data('789_ruby_5639.zip')))
bochan = list(token_generator(read_data('752_ruby_2438.zip')))

# sg=1: Skip-gram, sg=0: CBOW
model = Word2Vec([wagahaiwa + bochan], size=10, window=5, min_count=2, workers=2, sg=0)
vocab = model.wv.vocab
input_text = '吾輩は犬である'
print('入力文字: ' + input_text)

def create_mean_vector(token_list):
    return np.mean(np.array([
        model.wv[token]
        for token in token_list
        if vocab.get(token)
    ]), axis=0)

def pseudo_cosine_similarity(vector1, vector2):
    return np.dot(vector1, vector2)

input_mean_vector = create_mean_vector(list(token_generator(input_text)))

corpus_name_list = [
    '吾輩は猫である',
    '坊っちゃん',
]
print('候補の小説:')
print(corpus_name_list)

wagahaiwa_mean_vector = create_mean_vector(wagahaiwa)
bochan_mean_vector = create_mean_vector(bochan)

print(corpus_name_list[
    np.argmax([
        pseudo_cosine_similarity(input_mean_vector, wagahaiwa_mean_vector),
        pseudo_cosine_similarity(input_mean_vector, bochan_mean_vector),
    ])
])
