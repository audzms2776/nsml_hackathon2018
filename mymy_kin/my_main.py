# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
# Densely Connected Bidirectional LSTM with Applications to Sentence Classification

import argparse

import tensorflow as tf

from mymy_kin.dataset import KinQueryDataset, preprocess
from tensorflow.contrib import rnn

DATASET_PATH = './'


def _batch_loader(iterable, n=1):
    """
    데이터를 배치 사이즈만큼 잘라서 보내주는 함수입니다. PyTorch의 DataLoader와 같은 역할을 합니다

    :param iterable: 데이터 list, 혹은 다른 포맷
    :param n: 배치 사이즈
    :return:
    """
    length = len(iterable)
    for n_idx in range(0, length, n):
        yield iterable[n_idx:min(n_idx + n, length)]


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=500)
    args.add_argument('--batch', type=int, default=64)
    args.add_argument('--strmaxlen', type=int, default=400)
    args.add_argument('--embedding', type=int, default=8)
    args.add_argument('--threshold', type=float, default=0.5)
    config = args.parse_args()

    # 모델의 specification
    input_size = config.embedding * config.strmaxlen
    learning_rate = 0.001
    character_size = 251

    x = tf.placeholder(tf.int32, [None, config.strmaxlen])
    y_ = tf.placeholder(tf.float32, [None, 1])
    # 임베딩
    char_embedding = tf.get_variable('char_embedding', [character_size, config.embedding])
    embedded = tf.nn.embedding_lookup(char_embedding, x)

    with tf.variable_scope('layer1'):
        fw_lstm1 = rnn.BasicLSTMCell(num_units=8, activation=tf.nn.tanh, state_is_tuple=True)
        bw_lstm1 = rnn.BasicLSTMCell(num_units=8, activation=tf.nn.tanh, state_is_tuple=True)

        output, _ = tf.nn.bidirectional_dynamic_rnn(fw_lstm1, bw_lstm1, embedded, dtype=tf.float32)

        fw_out1 = output[0]
        bw_out1 = output[1]

    with tf.variable_scope('layer2'):
        h2 = tf.concat([embedded, fw_out1, bw_out1], axis=1)
        fw_lstm2 = rnn.BasicLSTMCell(num_units=8, activation=tf.nn.tanh, state_is_tuple=True)
        bw_lstm2 = rnn.BasicLSTMCell(num_units=8, activation=tf.nn.tanh, state_is_tuple=True)

        output2, _ = tf.nn.bidirectional_dynamic_rnn(fw_lstm2, bw_lstm2, h2, dtype=tf.float32)

        fw_out2 = output2[0]
        bw_out2 = output2[1]

    # flatten
    with tf.variable_scope('flatten'):
        h_ = tf.concat([fw_out2, bw_out2], axis=1)

        flat_layer = tf.contrib.layers.flatten(h_)
        output = tf.layers.dense(flat_layer, 1)

    sigmoid_output = tf.nn.sigmoid(output)

    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=y_))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    dataset = KinQueryDataset(DATASET_PATH, config.strmaxlen)
    dataset_len = len(dataset)
    one_batch_size = dataset_len // config.batch
    if dataset_len % config.batch != 0:
        one_batch_size += 1

    # epoch마다 학습을 수행합니다.
    for epoch in range(config.epochs):
        avg_loss = 0.0

        for i, (data, labels) in enumerate(_batch_loader(dataset, config.batch)):
            _, loss = sess.run([train_step, cross_entropy],
                               feed_dict={x: data, y_: labels})
            print('Batch : ', i + 1, '/', one_batch_size,
                  ', ERROR : ', float(loss))

            avg_loss += float(loss)

        print('epoch:', epoch, ' train_loss:', float(avg_loss / one_batch_size))
