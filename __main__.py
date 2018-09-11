import game
import algorithm
import tkinter
from time import sleep

try:
    winner, states = game.play_gui(algorithm.minimax(algorithm.basic_heuristic, 5))
    print(f'{(states[-1].board == -1).sum()} - {(states[-1].board == 1).sum()}')
    print({
        -1: 'Black Wins!',
        0: 'Draw!',
        1: 'White Wins!',
    }[winner])
    sleep(1)
except (tkinter.TclError, KeyboardInterrupt, EOFError):
    print('\nQuit')
    exit(1)