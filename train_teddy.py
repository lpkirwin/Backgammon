import numpy as np

# import matplotlib.pyplot as plt

import backgammon2
import pubeval2
import teddy

from tqdm import tqdm
import cProfile

# def plot_perf(performance):
#     plt.plot(performance)
#     plt.show()
#     return


def evaluate(agent, evaluation_agent, n_eval, n_games):
    wins = 0
    for i in range(n_eval):
        winner = backgammon2.play_game(agent, evaluation_agent)
        wins += int(winner == 1)
    win_rate = round(wins / n_eval * 100, 3)
    print(f"Win rate after training for {n_games} games: {win_rate}")
    return win_rate


def train(n_games=200_000, n_epochs=2_000, n_eval=1_000):

    agent = teddy
    evaluation_agent = None

    win_rates = []
    for g in tqdm(range(n_games)):

        if (g % n_epochs == 0) and (g != 0):
            win_rate = evaluate(agent, evaluation_agent, n_eval, n_games=g)
            win_rates.append(win_rate)

        _ = backgammon2.play_game(agent, agent, train=True)

    # plot_perf(winrates)


# ----- main -----
train()
# cProfile.run("train()")
