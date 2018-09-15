import game
import algorithm
import tqdm
import neural
import tqdm

white = 0
for i in tqdm.trange(100):
    winner, states = game.play(
        algorithm.stochastic_minimax(neural.heuristic, 2),
        algorithm.stochastic_minimax(algorithm.basic_heuristic, 2),
    )
    if winner == 1: white += 1
print(f'Winrate: {white}%')
