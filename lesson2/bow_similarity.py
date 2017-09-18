#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
from six.moves import urllib

from natsume_loader import token_generator, read_data
from gensim import corpora, matutils

import scipy.sparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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

# コーパスを作成
dictionary = corpora.Dictionary([
    wagahaiwa,
    bochan,
])
corpus = [dictionary.doc2bow(doc) for doc in [wagahaiwa, bochan]]

# 入力文字列との類似度を計算
input_text = '吾輩は犬である'
print('入力文字: ' + input_text)
input_bow = dictionary.doc2bow(list(token_generator(input_text)))

new_corpus = corpus + [input_bow]
corpus_name_list = [
    '吾輩は猫である',
    '坊っちゃん',
]
print('候補の小説:')
print(corpus_name_list)

scipy_csc_matrix = matutils.corpus2csc(new_corpus)
similarities = cosine_similarity(scipy_csc_matrix.transpose())

# 最も似ている小説の名前を出力
print('最も似ている小説: ' + corpus_name_list[np.argmax(similarities[2, :-1])])
