import othello
import tkinter as tk
from time import sleep
from threading import Thread


def play(p1, p2):
    game = othello.newgame()
    states = []
    while not game.winner():
        nodes = game.children()
        move = None
        while move not in nodes:
            if game.player == 1:
                move = p1(game)
            else:
                move = p2(game)
        states.append(game)
        game = move
    states.append(game)
    return game.winner(), states


def play_gui(algorithm):
    root = tk.Tk()
    root.title('Othello')
    canvas = tk.Canvas(root, height=641, width=641, bg='white')
    canvas.pack(fill=tk.BOTH, expand=True)
    root.resizable(False, False)

    def draw(state):
        w = 641
        h = 641
        canvas.create_rectangle(0, 0, w, h, fill='darkgreen')

        for i in range(3, w, 80):
            canvas.create_line([(i, 0), (i, h)], tag='grid_line')

        for i in range(3, h, 80):
            canvas.create_line([(0, i), (w, i)], tag='grid_line')

        for i, row in enumerate(state.board):
            for j, square in enumerate(row):
                if square:
                    canvas.create_oval(j * 80 + 10, i * 80 + 10, (j + 1) * 80 - 4, (i + 1) * 80 - 4,
                                       fill='black' if square == -1 else 'white')

    click_move = None

    def click(e):
        nonlocal click_move
        click_move = ((e.y - 3) // 80, (e.x - 3) // 80)

    game = othello.newgame()
    draw(game)
    canvas.bind("<Button-1>", click)
    root.update()

    game = othello.newgame()
    states = []
    while not game.winner():
        nodes = game.children()
        move = None
        while move not in nodes:
            if game.player == -1:
                print('Waiting for player input')
                while not click_move:
                    root.update()
                    sleep(0.05)
                move = game.place(click_move)
                if move not in nodes:
                    print(f'Invalid move {click_move}')
                click_move = None
            else:
                def run_algorithm():
                    nonlocal move
                    move = algorithm(game)
                print('Thinking')
                Thread(target=run_algorithm).start()
                sleep(1)
                while not move:
                    root.update()
                    sleep(0.05)
        states.append(game)
        game = move
        draw(game)
        root.update()
        print(game)
    states.append(game)
    return game.winner(), states
