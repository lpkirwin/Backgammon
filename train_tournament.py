import itertools
import time
import uuid
from collections import defaultdict
from copy import deepcopy

import numpy as np
import trueskill
from tqdm import tqdm

import backgammon2
import pubeval2
import teddy


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
    print(f"Evaluation results for {agent}:")
    print(wc)
    return wc


# IDS = list("abcdefghijklmnopqrstuvwxyz")


def new_id(s=""):
    # return IDS.pop(0)
    return s + uuid.uuid4().hex[:2]


def save_league(league):
    for agent in league:
        agent.save_model()


def train(league, anchor, n_rounds=1_000, n_games_each_pair=200, n_eval=1_000):

    max_league_size = 4

    env = trueskill.TrueSkill(draw_probability=0)
    for agent in league:
        agent.rating = env.create_rating()
    anchor.rating = env.create_rating()

    evaluation_agents = [None, anchor, pubeval2.PubEval()]
    eval_results = []

    league.append(anchor)

    for r in range(n_rounds):

        print()
        print(f"ROUND {r}")
        print("---------")
        print()

        matchups = list(itertools.combinations(league, 2)) * n_games_each_pair

        for p1, p2 in tqdm(matchups):

            # play a game
            winner, board = backgammon2.play_game(p1, p2, train=True)
            p1.game_over_update(board, winner == 1)
            p2.game_over_update(backgammon2.flip_board(board), winner == -1)

            # re-rank
            rating_groups = [(p1.rating,), (p2.rating,)]
            ranks = [0, 1] if winner == 1 else [1, 0]
            (p1.rating,), (p2.rating,) = env.rate(rating_groups, ranks=ranks)

        # sort league from best to worst, print out
        league = sorted(league, key=lambda a: a.rating.mu, reverse=True)
        max_name_length = max(len(str(a)) for a in league)
        for i, agent in enumerate(league):
            agent_name = str(agent)
            agent_name += " " * (max_name_length - len(agent_name))
            rank = (" " if len(str(i + 1)) == 1 else "") + str(i + 1)
            print(f"   {rank}. {agent_name}  {agent.rating.mu}")

        # evaluate best agent
        best_agent = league[1] if league[0] == anchor else league[0]
        eval_result = evaluate(best_agent, evaluation_agents, n_eval)
        eval_results.append(eval_result)

        # survival of the fittest
        if len(league) >= max_league_size:
            if league[-1] == anchor:
                to_drop = -2
            else:
                to_drop = -1
            print(f"dropping agent {league[to_drop]}")
            league.pop(to_drop)
        if league[0] == anchor:
            to_dupe = 1
        else:
            to_dupe = 0
        print(f"duplicating agent {league[to_dupe]}")
        new_agent = league[to_dupe].make_copy(name=new_id())
        new_agent.rating = deepcopy(league[to_dupe].rating)
        new_agent.save_model("cottage_current_best")
        new_agent.is_saveable = False
        league.append(new_agent)


if __name__ == "__main__":

    # vanilla_agents = [teddy.AgentTeddy(name=new_id("V")) for _ in range(3)]
    # rollout_agents = [
    #     teddy.AgentTeddy(name=new_id("R"), rollouts_during_training=True)
    #     for _ in range(3)
    # ]

    league = [
        teddy.AgentTeddy(saved_model="cottage_current_best", name="cottage") for i in range(1)
    ]
    league += [
        teddy.AgentTeddy(saved_model="38_thirtyeight", name="TT") for i in range(2)
    ]
    for agent in league:
        agent.is_saveable = False

    v4 = teddy.AgentTeddy(saved_model="38_thirtyeight", name="ANCHOR")
    v4.is_trainable = False

    # print()
    # print("vanilla agents")
    # print([str(a) for a in vanilla_agents])
    # print()
    # print("rollout agents")
    # print([str(a) for a in rollout_agents])

    # # league = vanilla_agents + rollout_agents
    # league = vanilla_agents

    train(league, anchor=v4)
