import numpy as np
from tqdm import tqdm
import time

import backgammon2
import pubeval2
import teddy
from collections import defaultdict


class WinCounter(object):
    def __init__(self):
        self.reset_memory()

    def reset_memory(self):
        self.memory = defaultdict(lambda: [0, 0])

    def store(self, player, won):
        self.memory[str(player)][0] += 1
        self.memory[str(player)][1] += int(won)

    def __str__(self):
        out = "Win rates per player:"
        for player in sorted(self.memory.keys()):
            games, wins = self.memory[player]
            out += f"\n    {player}: {wins / games:.2%} ({wins} out of {games} games)"
        return out


def evaluate(agent, opponents, n_eval):
    wc = WinCounter()
    for opp in opponents:
        if str(opp) in wc.memory.keys():
            continue
        for i in tqdm(range(n_eval)):
            winner, _ = backgammon2.play_game(agent, opp)
            wc.store(opp, winner == 1)
    print("Evaluation results:")
    print(wc)
    return wc


def train(agent, n_games=200_000, n_epochs=1_000, n_eval=200):

    # opponents = [None]  # + [pubeval2.PubEval()] * 2
    opponents = [agent]
    # max_opponents = 5
    evaluation_agents = [None, pubeval2.PubEval()]

    eval_results = []
    training_results = []
    wc = WinCounter()
    for g in tqdm(range(n_games)):

        if (g % n_epochs == 0) and (g > 0):

            # print training results
            print("Training results:")
            print(wc)
            training_results.append(wc)
            wc.reset_memory()

            # evaluate
            eval_result = evaluate(agent, evaluation_agents, n_eval)
            eval_results.append(eval_result)

            # # add current iteration to opponents
            # agent_copy = agent.make_copy()
            # agent_copy.is_trainable = False
            # opponents.append(agent_copy)
            # if len(opponents) > max_opponents:
            #     opponents.pop(0)

        opp = opponents[np.random.randint(len(opponents))]
        winner, board = backgammon2.play_game(agent, opp, train=True)
        agent.game_over_update(board, winner == 1)
        agent.game_over_update(backgammon2.flip_board(board), winner == -1)
        # wc.store(opp, winner == 1)

    # print(eval_results)


if __name__ == "__main__":

    # agent = teddy.AgentTeddy(saved_model="teddy_models/teddy_v1", model_name="teddy_v1")
    # agent = teddy.AgentTeddy(saved_model=None, model_name="teddy_v2")
    # agent = teddy.AgentTeddy(saved_model="teddy_models/teddy_v2", model_name="teddy_v3")
    agent = teddy.AgentTeddy(saved_model=None, model_name="teddy_v4")

    try:
        train(agent)
    except KeyboardInterrupt as e:
        print(e)
        agent.save_model()
