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


def train(agent, n_games=200_000, n_epochs=250, n_eval=1_000):

    # opponents = [None]  # + [pubeval2.PubEval()] * 2
    # opponents = [agent, pubeval2.PubEval()]
    # max_opponents = 5
    evaluation_agents = [None, pubeval2.PubEval()]

    eval_results = []
    training_results = []
    wc = WinCounter()

    best_random = 0.0
    best_pubeval = 0.0
    agent_copy = agent.make_copy()

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

            # gatekeep
            keep = True
            random_games, random_wins = eval_result.memory["None"]
            pubeval_games, pubeval_wins = eval_result.memory["pubeval"]
            random_rate = random_wins / random_games
            pubeval_rate = pubeval_wins / pubeval_games
            if random_rate < (best_random - 0.01):
                print("random win rate lower than best:", best_random)
                keep = False
            if pubeval_rate < (best_pubeval - 0.002):
                print("pubeval win rate lower than best:", best_pubeval)
                keep = False
            if not keep:
                print("EVALUATION FAILED, ROLLING BACK")
                agent = agent_copy
                agent.is_trainable = True
            else:
                print("EVALUATION PASSED")
                best_random = max(best_random, random_rate)
                best_pubeval = max(best_pubeval, pubeval_rate)

            agent_copy = agent.make_copy()
            agent_copy.is_trainable = False

            # # add current iteration to opponents
            # agent_copy.is_trainable = False
            # opponents.append(agent_copy)
            # if len(opponents) > max_opponents:
            #     opponents.pop(0)

        # opp = opponents[np.random.randint(len(opponents))]
        # winner, board = backgammon2.play_game(agent, opp, train=True)
        winner, board = backgammon2.play_game(agent, agent_copy, train=True)
        agent.game_over_update(board, winner == 1)
        # agent.game_over_update(backgammon2.flip_board(board), winner == -1)
        # wc.store(opp, winner == 1)

    # print(eval_results)


if __name__ == "__main__":

    # agent = teddy.AgentTeddy(saved_model="teddy_models/teddy_v1", model_name="teddy_v1")
    # agent = teddy.AgentTeddy(saved_model=None, model_name="teddy_v2")
    # agent = teddy.AgentTeddy(saved_model="teddy_models/teddy_v2", model_name="teddy_v3")
    # agent = teddy.AgentTeddy(saved_model=None, model_name="teddy_v4")
    # agent = teddy.AgentTeddy(saved_model="teddy_models/teddy_v4", model_name="teddy_v4")
    # ^ stuck at around 25%

    agent = teddy.AgentTeddy(saved_model=None, model_name="teddy_5G")

    try:
        train(agent)
    except KeyboardInterrupt as e:
        print(e)
        agent.save_model()
