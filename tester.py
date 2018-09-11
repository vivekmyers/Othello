import game
import algorithm
import tqdm

alg = algorithm.minimax(algorithm.basic_heuristic, 5)
winner, states = game.play(alg, alg)
print(winner)
