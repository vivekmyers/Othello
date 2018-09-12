import game
import algorithm
import tqdm
import neural
import tqdm

white = 0
for i in tqdm.trange(100):
    winner, states = game.play(
        algorithm.stochastic_minimax(neural.heuristic, 1),
        algorithm.stochastic,
    )
    if winner == 1: white += 1
print(f'Winrate: {white}%')
