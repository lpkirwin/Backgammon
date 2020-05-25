import numpy as np
import teddy
import backgammon2
import pubeval2
from tqdm import tqdm


class BoardCollector(object):
    def __init__(self, agent=None):
        self.boards = list()
        self.set_agent(agent)

    def set_agent(self, agent=None):
        self.agent = agent

    def action(self, board, board_array, **kwargs):
        self.boards.append(board_array)
        if self.agent is None:
            return np.random.randint(len(board_array))
        else:
            return self.agent.action(board, board_array, **kwargs)


if __name__ == "__main__":

    n_rounds = 2_000

    # generate data

    board_collector = BoardCollector()
    for _ in tqdm(range(n_rounds)):
        _ = backgammon2.play_game(board_collector, None)

    for _ in tqdm(range(n_rounds)):
        _ = backgammon2.play_game(board_collector, pubeval2)

    board_collector.set_agent(pubeval2)
    for _ in tqdm(range(n_rounds)):
        _ = backgammon2.play_game(board_collector, pubeval2)

    boards = np.concatenate(board_collector.boards, axis=0)
    np.random.shuffle(boards)
    print("Boards:", boards.shape)

    values = np.array([pubeval2.evaluate(b) for b in tqdm(boards)])
    values = values / 99999999.0  # pubeval max value
    print("Values:", values.shape)
    print("    mean:", np.mean(values))

    tmp = values.argsort()
    ranks = np.empty_like(tmp)
    ranks[tmp] = np.arange(len(values))
    ranks = ranks / ranks.max()
    print("Ranks:", ranks.shape)
    print("    mean:", np.mean(ranks))

    # train model with data

    agent = teddy.AgentTeddy()
    model = agent.model

    batch_size = 512
    n_batches = len(ranks) // 64
    n_epochs = 2

    loss = model.test_on_batch(boards, ranks)
    print("Loss:", loss)

    for _ in range(n_epochs):
        idx = 0
        for _ in tqdm(range(n_batches)):
            x = boards[idx : (idx + batch_size), :]  # noqa
            y = ranks[idx : (idx + batch_size)]  # noqa
            model.train_on_batch(x=x, y=y)
            idx += batch_size
        loss = model.test_on_batch(boards, ranks)
        print("Loss:", loss)

    # test resulting model

    n_games = 1_000
    wins = 0
    for _ in tqdm(range(n_games)):
        winner, _ = backgammon2.play_game(agent, pubeval2)
        if winner == 1:
            wins += 1
    print("Win rate:", wins / n_games)
