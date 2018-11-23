import game
import algorithm
import tqdm
import qlearn
import tqdm

white = 0
for i in tqdm.trange(100):
    winner, states = game.play(
        qlearn.maxq,
        algorithm.stochastic_minimax(algorithm.basic_heuristic, 1),
    )
    if winner == 1: white += 1
print(f'Winrate: {white}%')
