import game
import algorithm
import qlearn
import tkinter
from time import sleep

try:
    winner, states = game.play_gui(
        qlearn.maxq,
        delay=1,
    )
    sleep(1)
except (tkinter.TclError, KeyboardInterrupt, EOFError):
    print('\nQuit')
    exit(1)
