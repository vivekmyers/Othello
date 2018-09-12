import random
from sys import argv
import numpy as np
import algorithm
import game
import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from os import system, environ

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model_file = 'model/othello.ckpt'
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

input = tf.placeholder(tf.float32, shape=[None, 8, 8, 2], name='input')
conv1 = tf.layers.conv2d(
    inputs=input,
    filters=16,
    kernel_size=[5, 5],
    padding='same',
    activation=tf.nn.relu,
    name='conv1'
)
conv2 = tf.layers.conv2d(
    inputs=conv1,
    filters=32,
    kernel_size=[3, 3],
    padding='same',
    activation=tf.nn.relu,
    name='conv2'
)
conv_output = tf.reshape(conv2, shape=[-1, 32 * 8 * 8], name='reshape')
dense = tf.layers.dense(
    inputs=conv_output,
    units=32 * 8,
    name='dense1'
)
drop = tf.layers.dropout(
    inputs=dense,
    rate=0.5,
    name='dropout'
)
output = tf.layers.dense(
    inputs=drop,
    units=2,
    name='dense2'
)

prediction = tf.nn.softmax(output, name='prediction')
winner = tf.placeholder(tf.float32, shape=[None, 2], name='winner')
error = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=winner,
        logits=output,
        name='error',
    )
)
optimizer = tf.train.MomentumOptimizer(0.001, 0.9).minimize(error)
sess.run(tf.global_variables_initializer())

if __name__ == '__main__' and argv[1] == 'new':
    system('rm -rf model/*')
    saver = tf.train.Saver()
    saver.save(sess, model_file)
    print(f'New model saved to {model_file}')
    exit(0)

saver = tf.train.Saver()
saver.restore(sess, model_file)


def heuristic(state):
    board = game_tensor([state])
    out = sess.run(prediction, feed_dict={input: board})
    return out[0][0]


def benchmark():
    wincount = 0
    for i in tqdm.trange(100, desc='Benchmarking against greedy algorithm'):
        result, states = game.play(
            algorithm.stochastic_minimax(heuristic, 1),
            algorithm.stochastic_minimax(algorithm.basic_heuristic, 1),
        )
        if result == 1:
            wincount += 1
    return wincount


if __name__ == '__main__' and argv[1] != 'new':
    print(f'Winrate: {benchmark()}%')
    it = int(argv[1])
    for i in range(it):
        training_data = []
        testing_data = []
        print()
        for j in tqdm.trange(1000, desc='Simulating self play'):
            result, states = game.play(
                algorithm.stochastic_minimax(heuristic, 1),
                algorithm.stochastic_minimax(heuristic, 1),
            )
            for s in states:
                (training_data if j > 10 else testing_data).append([result, s])
        random.shuffle(training_data)
        print()


        def loss():
            inp = game_tensor([b for a, b in testing_data])
            correct_output = [[1, 0] if a == 1 else [0, 1] for a, b in testing_data]
            return sess.run(error, feed_dict={
                input: inp,
                winner: correct_output,
            })


        start = loss()
        end = start
        plt.figure()
        plt.axis([0, 1, 0, 1])
        prog = tqdm.trange(10000, desc='Training')
        for j in prog:
            batch = training_data[:]
            random.shuffle(batch)
            batch = batch[:16]
            inp = game_tensor([b for a, b in batch])
            correct_output = [[1, 0] if a == 1 else [0, 1] for a, b in batch]

            sess.run(optimizer, feed_dict={
                input: inp,
                winner: correct_output,
            })

            if random.random() < 0.1:
                inp = game_tensor([b for a, b in testing_data])
                correct_output = [[1, 0] if a == 1 else [0, 1] for a, b in testing_data]
                end = loss()
                prog.set_description("Training (loss={0:.5f})".format(end))
                prog.refresh()
                plt.scatter(j / 10000, end, s=20)
                plt.show()
                plt.pause(0.001)
        print()

        if end < start:
            saver.save(sess, model_file)
            print(f'Saving to {model_file}')
        else:
            saver.restore(sess, model_file)
            print('Model rejected, rolling back changes')
    print(f'Winrate: {benchmark()}%')
