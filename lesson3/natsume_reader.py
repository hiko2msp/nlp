#!/usr/bin/env python
# -*- coding:utf-8 -*-

from collections import Counter
import tensorflow as tf

def _read_words(filename):
    with open(filename) as f:
        return f.read().split()

def _build_vocab(token_list):
    counter = Counter(token_list)
    count_pairs = counter.most_common()
    words, _ = list(zip(*count_pairs))
    words = words[:10000]
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id

def _file_to_word_ids(token_list, word_to_id):
    return [word_to_id[word] for word in token_list if word in word_to_id]

def natsume_raw_data():
    train_path = 'wagahai.txt'
    valid_path = 'bochan.txt'
    test_path = 'kokoro.txt'

    word_to_id = _build_vocab(_read_words(train_path) + _read_words(valid_path))
    train_data = _file_to_word_ids(_read_words(train_path) + _read_words(valid_path), word_to_id)
    valid_data = _file_to_word_ids(_read_words(test_path), word_to_id)
    test_data = _file_to_word_ids(_read_words(test_path), word_to_id)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary

def natsume_producer(raw_data, batch_size, num_steps, name=None):
    '''
    夏目漱石のデータをイテレートします
    '''
    with tf.name_scope(name, 'NATSUMEProducer', [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size

        data = tf.reshape(raw_data[0 : batch_size * batch_len], [batch_size, batch_len])
        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps"
        )

        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")
        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps], [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1], [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        return x, y

if __name__ == '__main__':
    train_data, valid_data, test_data, _ = natsume_raw_data()
    x, y = natsume_producer(train_data, 10, 10)
    with tf.Session() as session:
        result = session.run(x)
    print(result)
