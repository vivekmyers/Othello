import game
import algorithm
import neural
import tkinter
from time import sleep

try:
    winner, states = game.play_gui(algorithm.stochastic_minimax(neural.heuristic, 2),
                                   algorithm.stochastic_minimax(algorithm.basic_heuristic, 2))
    sleep(1)
except (tkinter.TclError, KeyboardInterrupt, EOFError):
    print('\nQuit')
    exit(1)
