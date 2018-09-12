import game
import algorithm
import neural
import tkinter
from time import sleep

try:
    winner, states = game.play_gui(
        algorithm.minimax(neural.heuristic, 5),
        delay=1,
        heuristic=neural.heuristic,
    )
    sleep(1)
except (tkinter.TclError, KeyboardInterrupt, EOFError):
    print('\nQuit')
    exit(1)
