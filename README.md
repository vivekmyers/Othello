Plays Othello using a convolutional neural network heuristic function for minimax with alpha-beta pruning. Trains by simulating quick games with itself and using gradient descent to improve heristic board evaluations.

To play:
`python3 .`

To train:
`python3 neural.py [ITER]`

To clear model:
`python3 neural.py new`

To benchmark:
`python3 tester.py`

To show sample game against greedy minimax:
`python3 visual.py`

Dependencies:<br>
numpy<br>
tensorflow<br>
tqdm<br>
matplotlib<br>
