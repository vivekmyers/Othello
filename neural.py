import random
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
    units=2,
    name='dense2'
)

prediction = tf.nn.softmax(output, name='prediction')
winner = tf.placeholder(tf.float32, shape=[None, 2], name='winner')
error = tf.losses.get_regularization_loss() + tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=winner,
        logits=output,
        name='error',
    )
)
optimizer = tf.train.MomentumOptimizer(0.001, 0.9).minimize(error)
sess.run(tf.global_variables_initializer())

if __name__ == '__main__' and argv[1] == 'new':
    system('rm -rf model/othello*')
    saver = tf.train.Saver()
    saver.save(sess, model_file)
    print(f'New model saved to {model_file}')
    exit(0)

saver = tf.train.Saver()
saver.restore(sess, model_file)


def heuristic(state):
    if state.winner() is not None:
        return (state.winner() + 1) / 2
    board = game_tensor([state])
    out = sess.run(prediction, feed_dict={input: board})
    return out[0][0]


if __name__ == '__main__' and argv[1] != 'new':
    it = int(argv[1])

    for i in range(it):
        training_data = []
        testing_data = []
        for j in tqdm.trange(1000, desc='Simulating self play'):
            result, states = game.play(
                algorithm.stochastic_minimax(heuristic, 1),
                algorithm.stochastic_minimax(heuristic, 1),
            )
            for s in states:
                (training_data if j > 30 else testing_data).append([result, s])
        random.shuffle(training_data)
        print()


        def loss():
            inp = game_tensor([b for a, b in testing_data])
            correct_output = [[1, 0] if a == 1 else [0, 1] for a, b in testing_data]
            valid_loss = sess.run(error, feed_dict={
                input: inp,
                winner: correct_output,
            })
            inp = game_tensor([b for a, b in training_data])
            correct_output = [[1, 0] if a == 1 else [0, 1] for a, b in training_data]
            train_loss = sess.run(error, feed_dict={
                input: inp[:200],
                winner: correct_output[:200],
            })
            return valid_loss, train_loss
        
        start_loss, _ = loss()
        plt.figure()
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Time')
        plt.ylabel('Loss')
        plt.scatter(-1, 0, s=10, color='red', label="Training Loss")
        plt.scatter(-1, 0, s=10, color='blue', label="Validation Loss")
        plt.legend()
        prog = tqdm.trange(300, desc='Training')

        for j in prog:
            batch = random.sample(training_data, 256)

            sess.run(optimizer, feed_dict={
                input: game_tensor([b for a, b in batch]),
                winner: [[1, 0] if a == 1 else [0, 1] for a, b in batch]
            })

            valid_loss, train_loss = loss()
            prog.set_description("Training (loss={0:.5f})".format(valid_loss))
            prog.refresh()
            plt.scatter(j / 300, train_loss, s=5, color='red', label="Training Loss")
            plt.scatter(j / 300, valid_loss, s=5, color='blue', label="Validation Loss")
            plt.pause(0.001)

        print(f'Delta: {valid_loss - start_loss}')
        plt.close()
        print()
        saver.save(sess, model_file)
        print(f'Saving to {model_file}')
        print()
