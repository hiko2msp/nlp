#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
from natsume_reader import _build_vocab, _read_words
from create_natsume_data import token_generator
from natsume_word_lm import NATSUMEInput, MediumConfig

word_to_id = _build_vocab(_read_words('wagahai.txt') + _read_words('bochan.txt'))
id_to_word = {v:k for k, v in word_to_id.items()}
config = MediumConfig()

def get_model(input_data, reuse=False, num_steps=2):
    batch_size=1
    # モデルの準備
    with tf.device('/cpu:0'): 
        data_len = tf.size(input_data)
        batch_len = data_len
        data = tf.reshape(input_data[0 : batch_size * batch_len], [batch_size, batch_len])
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope("Model", reuse=reuse, initializer=initializer):

            embedding = tf.get_variable(
                "embedding", [config.vocab_size, config.hidden_size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, data)

            cell = tf.contrib.rnn.BasicLSTMCell(
                          config.hidden_size, forget_bias=0.0, state_is_tuple=True)
            cell = tf.contrib.rnn.MultiRNNCell(
                [cell for _ in range(config.num_layers)], state_is_tuple=True)
            
            state = cell.zero_state(batch_size, tf.float32)
    
            outputs = []
            with tf.variable_scope("RNN", reuse=reuse):
                for step in range(num_steps):
                    if step > 0: tf.get_variable_scope().reuse_variables()
                    (cell_output, state) = cell(inputs[:, step, :], state)

                    outputs.append(cell_output)
            output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])

            softmax_w = tf.get_variable(
                "softmax_w", [config.hidden_size, config.vocab_size], dtype=tf.float32)
            softmax_b = tf.get_variable("softmax_b", [config.vocab_size], dtype=tf.float32)

            logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
            logits = tf.reshape(logits, [batch_size, -1, config.vocab_size])
    return tf.reshape(tf.argmax(logits, axis=2), [-1])


def predict_next_word(session, input_text, data, reuse=False):
    input_token_list = list(token_generator(input_text))
    input_data = [word_to_id[word] for word in input_token_list if word in word_to_id]

    with tf.device('/cpu:0'): 
        # データの準備
        model = get_model(num_steps=len(input_data), reuse=reuse, input_data=data)

        token_id_list = session.run(model, 
            feed_dict={data: input_data}
        )
    
    print('input:', input_token_list)
    print('output:', [id_to_word[token_id] for token_id in token_id_list])
    try:
        return id_to_word[token_id_list[-1]]
    except:
        return '。'


def create_sentence(initial_text):
    data = tf.placeholder(tf.int32, shape=[None])
    model = get_model(data, reuse=False)
    saver = tf.train.Saver()
    with tf.Session() as session:
        ckpt = tf.train.get_checkpoint_state('./result')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            
            saver.restore(session, ckpt.model_checkpoint_path)

        sentense = initial_text
        next_word = ''

        # 。が生成された時に文章生成を止める
        while next_word != '。':
            next_word = predict_next_word(session, sentense, data, reuse=True)

            # 同じ文字が連続する場合は終了させる
            if sentense.endswith(next_word):
                sentense += '。'
                break
            sentense += next_word
    return sentense


if __name__ == '__main__':
    input_text = '今日は'
    print('初期テキスト:', input_text)
    print('生成された文章:', create_sentence(input_text))
