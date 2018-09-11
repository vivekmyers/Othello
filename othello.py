import numpy as np
from itertools import *
import functools


def newgame():
    board = np.zeros([8, 8])
    board[3, 3] = 1
    board[4, 4] = 1
    board[3, 4] = -1
    board[4, 3] = -1
    player = -1
    return State(board, player)


class State:
    def __init__(self, board, player):
        self.board = board
        self.player = player

    def __repr__(self):
        ret = '\n'
        for row in self.board:
            for square in row:
                ret += {1: 'o', 0: '.', -1: 'x'}[square] + ' '
            ret += '\n'
        return ret

    def __eq__(self, other):
        if type(other) != State:
            return False
        return self.player == other.player and (self.board == other.board).all()

    def __hash__(self):
        return hash(self.board.data) + hash(self.player) * 541

    def isvalid(self, move):
        x, y = move
        if self.board[(x, y)]:
            return False
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                idx = (x + i, y + j)
                if not (0 <= idx[0] < 8 and 0 <= idx[1] < 8):
                    continue
                if self.board[idx] == -self.player:
                    for k in count(1):
                        idx = (x + i * k, y + j * k)
                        if not (0 <= idx[0] < 8 and 0 <= idx[1] < 8):
                            break
                        val = self.board[idx]
                        if val == 0:
                            break
                        elif val == self.player:
                            return True
        return False

    def place(self, move):
        x, y = move
        board = np.copy(self.board)
        player = self.player
        if not (0 <= x < 8 and 0 <= y < 8) or board[(x, y)]:
            return State(board, -player)
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                idx = (x + i, y + j)
                if not (0 <= idx[0] < 8 and 0 <= idx[1] < 8):
                    continue
                if board[idx] == -player:
                    explored = []
                    for k in count(1):
                        idx = (x + i * k, y + j * k)
                        if not (0 <= idx[0] < 8 and 0 <= idx[1] < 8):
                            break
                        val = self.board[idx]
                        explored.append(idx)
                        if val == 0:
                            break
                        elif val == player:
                            for idx in explored:
                                board[idx] = player
                            board[x, y] = player
                            break
        return State(board, -player)

    def winner(self):
        def stuck(state):
            return not any(state.isvalid(move)
                           for move in product(*[range(x) for x in state.board.shape]))

        if all(chain(*self.board)) or (stuck(self) and stuck(self.children()[0])):
            return np.sign(self.board.sum())
        else:
            return 0

    def children(self):
        nodes = [self.place(move) for move in product(*[range(x)
                                                        for x in self.board.shape])
                 if self.isvalid(move)]
        if not nodes:
            nodes = [State(np.copy(self.board), -self.player)]
        return nodes
