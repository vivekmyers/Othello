import game
import algorithm
import tqdm
import neural
import tqdm

game.play_gui(
    algorithm.stochastic_minimax(neural.heuristic, 5),
    algorithm.stochastic_minimax(algorithm.basic_heuristic, 5),
    delay=1,
    heuristic=neural.heuristic,
)
