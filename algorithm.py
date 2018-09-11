from math import *

def basic_heuristic(state):
    return state.board.sum() + \
           state.board[([0, 0, 7, 7], [0, 7, 0, 7])].sum() * 4 + \
           state.board[0, :].sum() + state.board[:, 0].sum() +\
           state.board[7, :].sum() + state.board[:, 7].sum()


def greedy(state):
    return max(state.children(), key=lambda x: basic_heuristic(x) * state.player)


def human(state):
    print(state)
    nxt = state.place(eval('({0})'.format(input('> '))))
    if nxt in state.children():
        print(nxt)
    return nxt


def minimax(heuristic, max_depth):
    def evaluate(state, depth, alpha, beta):
        if depth == 1:
            return heuristic(state)
        elif state.player == 1:
            value = -inf
            for i in state.children():
                value = max(value, evaluate(i, depth - 1, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = inf
            for i in state.children():
                value = min(value, evaluate(i, depth - 1, alpha, beta))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value
    return lambda s: max(s.children(), key=lambda x: evaluate(x, max_depth, -inf, inf) * s.player)
