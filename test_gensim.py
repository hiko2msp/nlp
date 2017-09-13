#!/usr/bin/env python
# -*- coding:utf-8 -*-

from natsume_loader import token_generator, read_data
from gensim import corpora, matutils
import scipy.sparse
import numpy as np

wagahaiwa = list(token_generator(read_data('789_ruby_5639.zip')))
bochan = list(token_generator(read_data('752_ruby_2438.zip')))

dictionary = corpora.Dictionary([
    wagahaiwa,
    bochan,
])
dictionary.save('./natsume.dict')
corpus = [dictionary.doc2bow(doc) for doc in [wagahaiwa, bochan]]

from sklearn.metrics.pairwise import cosine_similarity
input_text = '吾輩は犬である'
print(list(token_generator(input_text)))
input_bow = dictionary.doc2bow(list(token_generator(input_text)))

new_corpus = corpus + [input_bow]

scipy_csc_matrix = matutils.corpus2csc(new_corpus)
similarities = cosine_similarity(scipy_csc_matrix.transpose())
print(similarities)

print(np.argmax(similarities[2, :-1]))
