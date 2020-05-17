import numpy as np

# import matplotlib.pyplot as plt
# import random_agent

# import sys
import time

# import pubeval
# import kotra

# import functools
from tqdm import tqdm
from numba import njit
from numba.typed import List

# Board layout (from Tesauro)
#                                         1j | 1o 2o
# 13 14 15 16 17 18 | 19 20 21 22 23 24 | 25 | 26 27 | 28 <- die counter
# 12 11 10 09 08 07 | 06 05 04 03 02 01 | 00 
#                                         2j
# 1j, 2j = jail (for p1 and p2)
# 1o, 2o = off the board


def init_board():
    board = np.zeros(29)
    board[1] = -2
    board[12] = -5
    board[17] = -3
    board[19] = -5
    board[6] = 5
    board[8] = 3
    board[13] = 5
    board[24] = 2
    return board


STARTING_BOARD = init_board()

FLIP_INDEX = np.array(list(range(25, -1, -1)) + [27, 26] + [28])


@njit()
def flip_board(board):
    return board[FLIP_INDEX].copy() * -1


flipped_starting_board = flip_board(STARTING_BOARD)
assert (STARTING_BOARD == flipped_starting_board).all()


dice_buffer = list()


def roll_dice(buffer_size=1_000_000):
    global dice_buffer
    try:
        return dice_buffer.pop()
    except IndexError:
        dice_buffer = list(np.random.randint(1, 7, size=(buffer_size, 2)))
        return dice_buffer.pop()


@njit()
def game_over(board):
    # returns True if the game is over
    return board[26] == 15 or board[27] == -15


@njit()
def check_for_error(board):

    error_in_game = False

    player_1_pieces = board[board > 0].sum()
    player_2_pieces = board[board < 0].sum()

    if (player_1_pieces != 15) or (player_2_pieces != -15):
        error_in_game = True

    return error_in_game


# def pretty_print(board):
#     string = str(
#         np.array2string(board[1:13])
#         + "\n"
#         + np.array2string(board[24:12:-1])
#         + "\n"
#         + np.array2string(board[25:29])
#     )
#     print("board: \n", string)


@njit()
def make_move_container():
    return list([empty_move()])


@njit()
def add_move_to_container(container, move):
    container.append(move)
    # container.union([move])


@njit()
def empty_move():
    return np.zeros(29)


@njit()
def convert_to_move(a, b, capture=False):
    move = empty_move()
    move[a] -= 1
    move[b] += 1
    if capture:
        move[b] += 1
        move[0] -= 1
    move[28] = 1  # counter for number of dice used
    return move


@njit()
def valid_endpoint(board, x):
    if board[x] < 5:
        if board[x] > -2:
            if x > 0:
                return True
    return False


@njit()
def captured_endpoint(board, x):
    return board[x] == -1


@njit()
def moves_for_one_die(board, die):

    # board = board.copy()

    # moves = make_move_container()
    moves = list([empty_move()])

    is_game_over = game_over(board)
    if is_game_over:
        # add_move_to_container(moves, empty_move())
        return moves

    # if you're in jail
    if board[25] > 0:
        a = 25
        b = a - die
        is_valid = valid_endpoint(board, b)
        is_captured = captured_endpoint(board, b)
        if is_valid:
            move = convert_to_move(a, b, capture=is_captured)
            # add_move_to_container(moves, move)
            moves.append(move)

    # else not in jail
    else:

        is_bearing_off = (board[7:25] > 0).sum() == 0
        if is_bearing_off:

            max_position = np.where(board[0:7] > 0)[0].max()

            # if there are pieces exactly on the die value
            if board[die] > 0:
                a = die
                b = 26
                move = convert_to_move(a, b)
                # add_move_to_container(moves, move)
                moves.append(move)

            # if all pieces are past the die value
            elif die > max_position:
                a = max_position
                b = 26
                move = convert_to_move(a, b)
                # add_move_to_container(moves, move)
                moves.append(move)

        # add all regular moves
        current_positions = np.where(board[0:25] > 0)[0]
        for a in current_positions:
            b = a - die
            is_valid = valid_endpoint(board, b)
            is_captured = captured_endpoint(board, b)
            if is_valid:
                move = convert_to_move(a, b, capture=is_captured)
                # add_move_to_container(moves, move)
                moves.append(move)

    # # if you can't do anything, skip turn
    # move_count = len(moves)
    # if not move_count:
    #     add_move_to_container(moves, empty_move())

    return moves


@njit()
def moves_for_two_dice(board, dice):

    # assert board[0] <= 0
    # assert board[25] >= 0
    # assert board[26] >= 0
    # assert board[27] <= 0
    # assert board[28] == 0

    moves = make_move_container()
    # moves = list()

    # doubles = dice[0] == dice[1]
    doubles = False

    if doubles:

        first_moves = moves_for_one_die(board, dice[0])
        for m1 in first_moves:
            tmp_board = board + m1
            second_moves = moves_for_one_die(tmp_board, dice[0])
            for m2 in second_moves:
                tmp_board = board + m1 + m2
                third_moves = moves_for_one_die(tmp_board, dice[0])
                for m3 in third_moves:
                    tmp_board = board + m1 + m2 + m3
                    fourth_moves = moves_for_one_die(tmp_board, dice[0])
                    for m4 in fourth_moves:
                        move = m1 + m2 + m3 + m4
                        add_move_to_container(moves, move)

    else:

        first_moves = moves_for_one_die(board, dice[0])
        for m1 in first_moves:
            tmp_board = board + m1
            second_moves = moves_for_one_die(tmp_board, dice[1])
            for m2 in second_moves:
                move = m1 + m2
                add_move_to_container(moves, move)

        # same again, but with second die first
        first_moves = moves_for_one_die(board, dice[1])
        for m1 in first_moves:
            tmp_board = board + m1
            second_moves = moves_for_one_die(tmp_board, dice[0])
            for m2 in second_moves:
                move = m1 + m2
                add_move_to_container(moves, move)

    moves = manual_concat(moves)

    return moves


@njit()
def manual_concat(arrays):

    n_rows = len(arrays)
    n_cols = len(arrays[0])
    new_array = np.zeros(shape=(n_rows, n_cols))

    i = 0
    for x in arrays:

        skip = False

        # # check for uniqueness of moves (a bit slow)
        # for j in range(i):
        #     if (x == new_array[j, :]).all():
        #         skip = True

        if not skip:
            new_array[i, :] = x
            i += 1


    # have to use all your moves if you're able to, so need
    # to filter out moves that don't use the maximum number
    # of dice
    max_dice_used = new_array[:, 28].max()
    new_array = new_array[new_array[:, 28] == max_dice_used, :]
    new_array[:, 28] = 0

    return new_array


# def process_array(move_array, dice):
#     # have to use all your moves if you're able to, so need
#     # to filter out moves that don't use the maximum number
#     # of dice
#     # move_array = np.unique(move_array, axis=0)
#     if move_array.ndim > 1:
#         doubles = dice[0] == dice[1]
#         dice_available = 4 if doubles else 2
#         n_dice_used = move_array[move_array]
#         n_dice_used = (np.abs(move_array).sum(axis=1) / 2).clip(max=dice_available)
#         max_dice_used = np.max(n_dice_used)
#         move_array = move_array[n_dice_used == max_dice_used]
#     return move_array


class RandomPlayer(object):
    def action(self, board, board_array, **kwargs):
        return np.random.randint(len(board_array))


def play_game(player1=None, player2=None, train=False):

    if player1 is None:
        player1 = RandomPlayer()
    if player2 is None:
        player2 = RandomPlayer()

    board = STARTING_BOARD.copy()
    player = np.random.choice([1, -1])  # which player begins?
    turn = 0

    while True:

        dice = roll_dice()
        doubles = dice[0] == dice[1]

        for _ in range(int(doubles) + 1):

            # moves = moves_for_two_dice(board, dice)
            move_array = moves_for_two_dice(board, dice)
            # move_array = process_array(move_array, dice)

            board_array = board + move_array

            if player == 1:
                action = player1.action(board, board_array, train=train)
                chosen_move = move_array[action]
            else:
                action = player2.action(board, board_array, train=train)
                chosen_move = move_array[action]

            board = board + chosen_move

            # check exit conditions
            # game_has_error = check_for_error(board)
            # if game_has_error:
            #     raise ValueError("Game in error state")
            is_game_over = game_over(board)
            if is_game_over:
                return player, board

        # prep for next turn
        player *= -1
        board = flip_board(board)
        turn += 1

        # if turn % 1000 == 0:
        #     print("check in")

    # return the winner
    return -1 * player, board


def plot_perf(performance):
    plt.plot(performance)
    plt.show()
    return


def log_status(g, wins, performance, nEpochs):
    if g == 0:
        return performance
    print("game number", g)
    win_rate = wins / nEpochs
    print("win rate:", win_rate)
    performance.append(win_rate)
    return performance


def main():
    startTime = time.time()
    winners = {}
    winners["1"] = 0
    winners["-1"] = 0
    # Collecting stats of the games
    nGames = 5000  # how many games?
    performance = list()
    player1 = None
    player2 = None
    wins = 0
    nEpochs = 1_000
    print("Playing " + str(nGames) + " between" + str(player1) + " and " + str(player2))
    for g in tqdm(range(nGames)):
        if g % nEpochs == 0:
            performance = log_status(g, wins, performance, nEpochs)
            wins = 0
        winner, _ = play_game(player1, player2)
        winners[str(winner)] += 1
        wins += winner == 1
    print("Out of", nGames, "games,")
    print("player", 1, "won", winners["1"], "times and")
    print("player", -1, "won", winners["-1"], "times")
    runTime = time.time() - startTime
    print("runTime:", runTime)
    print("average time:", runTime / nGames)
    # plot_perf(performance)


if __name__ == "__main__":
    import cProfile

    # cProfile.run("main()")
    main()
