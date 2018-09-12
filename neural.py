import random
from sys import argv
import numpy as np
import algorithm
import game
import tqdm
import tensorflow as tf
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
optimizer = tf.train.MomentumOptimizer(0.0005, 0.5).minimize(error)
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
    bench = benchmark()
    print(f'Winrate: {bench}%')
    it = int(argv[1])
    for i in range(it):
        training_data = []
        print()
        for j in tqdm.trange(100, desc='Simulating self play'):
            result, states = game.play(
                algorithm.stochastic_minimax(heuristic, 1),
                algorithm.stochastic_minimax(heuristic, 1),
            )
            for s in states:
                training_data.append([result, s])
        random.shuffle(training_data)
        testing_data = training_data[:10]
        training_data = training_data[10:]
        print()
        prog = tqdm.trange(1000, desc='Training')
        for j in prog:
            batch = training_data[:]
            random.shuffle(batch)
            batch = batch[:10]
            inp = game_tensor([b for a, b in batch])
            correct_output = [[1, 0] if a == 1 else [0, 1] for a, b in batch]

            sess.run(optimizer, feed_dict={
                input: inp,
                winner: correct_output,
            })

            if random.random() < 0.3:
                inp = game_tensor([b for a, b in testing_data])
                correct_output = [[1, 0] if a == 1 else [0, 1] for a, b in testing_data]
                prog.set_description("Training (loss={0:.5f})".format(
                    sess.run(error, feed_dict={
                        input: inp,
                        winner: correct_output,
                    })))
                prog.refresh()
        print()
        strength = benchmark()
        print(f'Winrate: {strength}%')
        if strength >= bench:
            bench = strength
            saver.save(sess, model_file)
            print()
            print(f'Saving to {model_file}')
        else:
            saver.restore(sess, model_file)
            print()
            print('Model rejected, rolling back changes')
