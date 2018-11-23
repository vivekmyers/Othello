import random
from math import *
from sys import argv
import numpy as np
import algorithm
import game
import tqdm
import tensorflow as tf
from sys import platform

if platform == 'darwin':
    import matplotlib

    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from os import system, environ

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model_file = 'model/qlearn.ckpt'
tf.logging.set_verbosity(tf.logging.ERROR)


def game_tensor(states):
    boards = [state.board.astype(np.float32) for state in states]
    players = [np.full(state.board.shape,
                       np.float32(state.player)) for state in states]
    return np.array([np.stack([board, player], axis=2)
                     for board, player in zip(boards, players)])


def inverse_game_tensor(states):
    boards = [-state.board.astype(np.float32) for state in states]
    players = [np.full(state.board.shape,
                       np.float32(-state.player)) for state in states]
    return np.array([np.stack([board, player], axis=2)
                     for board, player in zip(boards, players)])


sess = tf.InteractiveSession()

input = tf.placeholder(tf.float32, shape=[None, 8, 8, 4], name='input')
conv1 = tf.layers.conv2d(
    inputs=input,
    filters=16,
    kernel_size=[5, 5],
    padding='same',
    activation=tf.nn.relu,
    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.001),
    name='conv1'
)
conv2 = tf.layers.conv2d(
    inputs=conv1,
    filters=32,
    kernel_size=[3, 3],
    padding='same',
    activation=tf.nn.relu,
    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.001),
    name='conv2'
)
conv3 = tf.layers.conv2d(
    inputs=conv2,
    filters=32,
    kernel_size=[3, 3],
    padding='same',
    activation=tf.nn.relu,
    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.001),
    name='conv3'
)
conv_output = tf.reshape(conv3, shape=[-1, 32 * 8 * 8], name='reshape')
dense = tf.layers.dense(
    inputs=conv_output,
    units=32 * 8 * 2,
    name='dense1'
)
drop = tf.layers.dropout(
    inputs=dense,
    rate=0.5,
    name='dropout'
)
output = tf.layers.dense(
    inputs=drop,
    units=1,
    name='dense2'
)

computed = tf.placeholder(tf.float32, shape=[None, 1], name='computed')
error = tf.losses.get_regularization_loss() + tf.reduce_mean(tf.squared_difference(computed, output))

epsilon = 0.1
alpha = 0.0005
gamma = 0.95

optimizer = tf.train.AdamOptimizer(alpha).minimize(error)
sess.run(tf.global_variables_initializer())

if __name__ == '__main__' and argv[1] == 'new':
    system('rm -rf model/qlearn*')
    saver = tf.train.Saver()
    saver.save(sess, model_file)
    print(f'New model saved to {model_file}')
    exit(0)

saver = tf.train.Saver()
saver.restore(sess, model_file)


def stochastic_maxq(s):
    if random.random() < epsilon:
        return random.choice(s.children())
    else:
        return maxq(s)


def maxq(s):
    return max(s.children(), key=lambda a: Q(s, a))


def R(s, a, s0):
    if s0.winner() == s.player:
        return 1000
    elif s0.winner() == -s.player:
        return -1000
    else:
        return (s0.board - s.board).sum() * s.player


def train(s, a, q):
    board1 = game_tensor([s])
    board2 = game_tensor([a])
    tensor = np.concatenate([board1, board2], axis=3)
    out = sess.run(optimizer, feed_dict={input: tensor, computed: np.array([[q]])})


def Q(s, a):
    board1 = game_tensor([s])
    board2 = game_tensor([a])
    tensor = np.concatenate([board1, board2], axis=3)
    out = sess.run(output, feed_dict={input: tensor})
    return out[0][0]
    

if __name__ == '__main__' and argv[1] != 'new':
    it = int(argv[1])
    
    train_tuples = []
    for i in range(it):
        for j in tqdm.trange(1000, desc='Simulating self play'):
            result, states = game.play(
                stochastic_maxq,
                stochastic_maxq,
            )
            states += [states[-1].children()[0], states[-1].children()[0].children()[0]]
            for i in range(len(states) - 2):
                s, a, s0 = states[i:(i+3)]
                q0 = R(s, a, s0) + gamma * max(Q(s0, a0) for a0 in s0.children())
                random.shuffle(train_tuples)
                train_tuples += [(s, a, q0)]
                if len(train_tuples) > 500:
                    train_tuples = train_tuples[1:]
            for i in train_tuples[:50]:
                train(*i)

        saver.save(sess, model_file)
        white = 0
        for i in tqdm.trange(100, desc='Benchmarking'):
            winner, states = game.play(
                    maxq,
                    algorithm.stochastic_minimax(algorithm.basic_heuristic, 1),
                    )
            if winner == 1: 
                white += 1
        print(f'Winrate: {white}%')

        print(f'Saving to {model_file}')
        print()

