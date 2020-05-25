import numpy as np
from numba import float32, int32, jitclass

import backgammon2

DICE = (
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
    (2, 6),
    (3, 3),
    (3, 4),
    (3, 5),
    (3, 6),
    (4, 4),
    (4, 5),
    (4, 6),
    (5, 5),
    (5, 6),
    (6, 6),
)

spec = [
    ("n_boards", int32),
    ("data_size", int32),
    ("boards", float32[:, :]),
    ("board_idxs", float32[:]),
    ("dice_1", float32[:]),
    ("dice_2", float32[:]),
    ("results", float32[:]),
    ("ptr", int32),
    ("max_ptr", int32),
    ("dice", int32[:, :]),
]


@jitclass(spec)
class RolloutManager(object):
    def __init__(self, n_boards=32, data_size=500_000):
        self.n_boards = n_boards
        self.data_size = data_size
        self.boards = np.zeros(shape=(data_size, 29), dtype=np.float32)
        self.board_idxs = np.zeros(shape=(data_size,), dtype=np.float32)
        self.dice_1 = np.zeros(shape=(data_size,), dtype=np.float32)
        self.dice_2 = np.zeros(shape=(data_size,), dtype=np.float32)
        self.results = np.zeros(shape=(n_boards,), dtype=np.float32)
        self.ptr = 0
        self.max_ptr = 0
        self.dice = np.array(DICE, dtype=np.int32)

    def full_rollout_boards(self, board_array):

        self.ptr = 0

        board_array = backgammon2.flip_board_array(board_array)

        for board_idx, board in enumerate(board_array):

            for d1, d2 in self.dice:

                new_moves = backgammon2.moves_for_two_dice(
                    board, d1, d2, all_doubles=True, make_unique=False
                )

                new_boards = board + new_moves
                new_ptr = self.ptr + len(new_boards)
                self.boards[self.ptr : new_ptr] = new_boards
                self.board_idxs[self.ptr : new_ptr] = board_idx
                self.dice_1[self.ptr : new_ptr] = d1
                self.dice_2[self.ptr : new_ptr] = d2
                self.ptr = new_ptr

        self.max_ptr = max(self.ptr, self.max_ptr)

        return backgammon2.flip_board_array(self.boards[: self.ptr])

    def maxmin_rollout_value(self, values):

        assert values.shape == (self.ptr,)

        # vars with underscores represent last index's vars
        board_idx_, d1_, d2_ = 0, 1, 1
        current_min_value = np.nan
        board_value = 0.0

        for idx in range(self.ptr):

            value = values[idx]
            board_idx = self.board_idxs[idx]
            d1 = self.dice_1[idx]
            d2 = self.dice_2[idx]

            # 0, 1, 1
            # 0, 1, 1
            # 0, 1, 2
            # 0, 1, 3
            # 0, 1, 3
            # ...

            # if same set of dice, keep track of cummin value
            if (d1 == d1_) and (d2 == d2_):
                current_min_value = min(value, current_min_value)
            # if next dice, then add min value to board value
            else:
                prb = 1 / 36 if (d1_ == d2_) else 2 / 36
                board_value += current_min_value * prb
                current_min_value = value

            # if done board, store result for board
            if board_idx != board_idx_:
                self.results[int(board_idx_)] = np.float32(board_value)
                board_value = 0

            board_idx_, d1_, d2_ = board_idx, d1, d2

        # store result for final board
        self.results[int(board_idx_)] = np.float32(board_value)

        return self.results[: board_idx_ + 1]


if __name__ == "__main__":

    rm = RolloutManager()
    boards = np.stack([backgammon2.STARTING_BOARD for _ in range(32)])
    rollout_boards = rm.full_rollout_boards(boards)
    rollout_values = np.random.uniform(size=(len(rollout_boards,))).astype(np.float32)
    board_values = rm.maxmin_rollout_value(rollout_values)
    print(board_values)
