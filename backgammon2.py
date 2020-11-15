
import time

import numpy as np
from numba import njit
from tqdm import tqdm

import pubeval2
import teddy

# Board layout (from Tesauro)
#                                         1j | 1o 2o
# 13 14 15 16 17 18 | 19 20 21 22 23 24 | 25 | 26 27 | 28 <- die counter
# 12 11 10 09 08 07 | 06 05 04 03 02 01 | 00
#                                         2j
# 1j, 2j = jail (for p1 and p2)
# 1o, 2o = off the board


def init_board():
    board = np.zeros(29, dtype=np.float32)
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

FLIP_INDEX = np.array(list(range(25, -1, -1)) + [27, 26] + [28], dtype=np.int32)


@njit()
def flip_board(board):
    flipped_board = board[FLIP_INDEX].copy() * -1
    return flipped_board


@njit()
def flip_board_array(board_array):
    flipped_board_array = board_array[:, FLIP_INDEX].copy() * -1
    return flipped_board_array


@njit()
def reset_move_counter(board):
    board[28] = 0
    return board


flipped_starting_board = flip_board(STARTING_BOARD)
assert (STARTING_BOARD == flipped_starting_board).all()


dice_buffer = list()


def roll_dice(buffer_size=100_000):
    global dice_buffer
    try:
        return dice_buffer.pop()
    except IndexError:
        dice_buffer = list(np.random.randint(1, 7, size=(buffer_size, 2)))
        return dice_buffer.pop()


@njit()
def game_over(board):
    return board[26] == 15 or board[27] == -15


@njit()
def game_over_array(board_array):
    return (board_array[:, 26] == 15) | (board_array[:, 27] == -15)


@njit()
def check_for_error(board):

    errors = ""

    player_1_pieces = board[board > 0].sum()
    player_2_pieces = board[board < 0].sum()
    if (player_1_pieces != 15) or (player_2_pieces != -15):
        errors += "wrong number of pieces - "

    if board[0] > 0:
        errors += "positive value in p2 jail - "
    if board[25] < 0:
        errors += "negative value in p1 jail - "
    if board[26] < 0:
        errors += "negative value in p1 off board - "
    if board[27] > 0:
        errors += "positive value in p2 off board - "
    if board[28] != 0:
        errors += "move counter nonzero - "

    return errors


POSITION_NAMES = (
    ["P2 jail"]
    + list("ABCDEFGHIJKLMNOPQRSTUVWX")
    + ["P1 jail", "P1 off", "P2 off", "_move_count"]
)


def name_to_index(name, player):
    if player == -1:
        pos_names = list(np.array(POSITION_NAMES)[FLIP_INDEX])
    else:
        pos_names = POSITION_NAMES
    if name in pos_names:
        return pos_names.index(name)
    else:
        raise ValueError("Invalid name")


def print_move(move, player):
    if player == -1:
        move = flip_board(move) * -1
    for i, x in enumerate(move):
        name = POSITION_NAMES[i]
        if x > 0:
            print(f"{name}: {''.join('+' for _ in range(int(x)))}")
        if x < 0:
            print(f"{name}: {''.join('-' for _ in range(abs(int(x))))}")


def input_dice():
    dice = None
    while dice is None:
        try:
            dice = input("Enter dice roll: ")
            dice = make_dice(int(dice[0]), int(dice[1]))
        except Exception as e:
            print("Invalid dice!", e)
            dice = None
    return dice


@njit()
def make_dice(a, b):
    # make dice that numba won't reject
    return np.array([a, b], dtype=np.int32)


def print_dice(dice):
    print("Dice:", " ".join(f"[{d}]" for d in dice))


def print_player(player):
    print("Player:", "1 (X)" if player == 1 else "2 (O)")


def input_move(board, player, dice):

    doubles = dice[0] == dice[1]
    dice_list = list(dice.copy())
    if doubles:
        dice_list = dice_list * 2
    n_dice = len(dice_list)
    i = 1

    print_board(board, player)
    print_player(player)
    print_dice(dice_list)

    while dice_list:

        legal_moves = list()
        for die in dice_list:
            legal_moves.extend(moves_for_one_die(board, die))
        legal_move_array = np.stack(legal_moves)
        legal_move_array = filter_moves(legal_move_array)

        if legal_move_array.sum() == 0:
            print("No legal moves?")

        move_string = input(f">> Enter move {i}: ")

        # if no input, ask again
        if not move_string:
            continue

        # convention for picking random move
        if move_string == " ":
            if len(dice_list) != n_dice:
                print("Making one random move (or no move)")
                board_array = board + legal_move_array
                action = RandomPlayer().action(board, board_array)
                chosen_move = legal_move_array[action]
                board = board + chosen_move
                print_move(chosen_move, player)
            else:
                print(f"Making {n_dice} random moves (or no move)")
                for _ in range(int(doubles) + 1):
                    move_array = moves_for_two_dice(board, dice[0], dice[1])
                    board_array = board + move_array
                    action = RandomPlayer().action(board, board_array)
                    chosen_move = move_array[action]
                    board = board + chosen_move
                    print_move(chosen_move, player)
            return board

        # if in jail then only input number, if bearing off
        # then only input position
        if len(move_string) == 1:
            try:
                a = 25
                delta = int(move_string)
                b = a - delta
            except ValueError:
                try:
                    a = name_to_index(move_string.upper(), player)
                    b = 26
                    # use smallest eligible dice
                    delta = min([d for d in dice_list if d >= a])
                except Exception:
                    is_valid = False

        elif len(move_string) > 1:
            try:
                a = name_to_index(move_string[0].upper(), player)
                delta = int(move_string[1])
                b = a - delta
            except Exception:
                is_valid = False

        try:
            is_captured = captured_endpoint(board, b)
            move = convert_to_move(a, b, capture=is_captured)
            is_valid = any([(move == m).all() for m in list(legal_move_array)])
        except Exception:
            is_valid = False

        if is_valid:
            board = board + move
            board = reset_move_counter(board)
            dice_list.pop(dice_list.index(delta))
            i += 1

            if is_captured:
                print("Nice capture :)")

            print_board(board, player)
            print_player(player)
            print_dice(dice_list)

        else:
            print("Invalid move!")

    print("Done, thanks!")
    return board


def print_board(board, player):

    # always show from player 1's perspective
    if player == -1:
        board = flip_board(board)

    def p(x, row):
        x = board[x]
        if abs(x) > row:
            p = "X" if x > 0 else "O"
        else:
            p = "_"
        return p

    out = f"""
    P2 off: {"".join("O" for _ in range(int(abs(board[27]))))}
    P2 jail: {"".join("O" for _ in range(int(abs(board[0]))))}

      M N O P Q R | S T U V W X
    | {p(13, 0)} {p(14, 0)} {p(15, 0)} {p(16, 0)} {p(17, 0)} {p(18, 0)} | {p(19, 0)} {p(20, 0)} {p(21, 0)} {p(22, 0)} {p(23, 0)} {p(24, 0)} |
    | {p(13, 1)} {p(14, 1)} {p(15, 1)} {p(16, 1)} {p(17, 1)} {p(18, 1)} | {p(19, 1)} {p(20, 1)} {p(21, 1)} {p(22, 1)} {p(23, 1)} {p(24, 1)} |
    | {p(13, 2)} {p(14, 2)} {p(15, 2)} {p(16, 2)} {p(17, 2)} {p(18, 2)} | {p(19, 2)} {p(20, 2)} {p(21, 2)} {p(22, 2)} {p(23, 2)} {p(24, 2)} |
    | {p(13, 3)} {p(14, 3)} {p(15, 3)} {p(16, 3)} {p(17, 3)} {p(18, 3)} | {p(19, 3)} {p(20, 3)} {p(21, 3)} {p(22, 3)} {p(23, 3)} {p(24, 3)} |
    | {p(13, 4)} {p(14, 4)} {p(15, 4)} {p(16, 4)} {p(17, 4)} {p(18, 4)} | {p(19, 4)} {p(20, 4)} {p(21, 4)} {p(22, 4)} {p(23, 4)} {p(24, 4)} |
    |             |             |
    | {p(12, 4)} {p(11, 4)} {p(10, 4)} {p( 9, 4)} {p( 8, 4)} {p( 7, 4)} | {p( 6, 4)} {p( 5, 4)} {p( 4, 4)} {p( 3, 4)} {p( 2, 4)} {p( 1, 4)} |
    | {p(12, 3)} {p(11, 3)} {p(10, 3)} {p( 9, 3)} {p( 8, 3)} {p( 7, 3)} | {p( 6, 3)} {p( 5, 3)} {p( 4, 3)} {p( 3, 3)} {p( 2, 3)} {p( 1, 3)} |
    | {p(12, 2)} {p(11, 2)} {p(10, 2)} {p( 9, 2)} {p( 8, 2)} {p( 7, 2)} | {p( 6, 2)} {p( 5, 2)} {p( 4, 2)} {p( 3, 2)} {p( 2, 2)} {p( 1, 2)} |
    | {p(12, 1)} {p(11, 1)} {p(10, 1)} {p( 9, 1)} {p( 8, 1)} {p( 7, 1)} | {p( 6, 1)} {p( 5, 1)} {p( 4, 1)} {p( 3, 1)} {p( 2, 1)} {p( 1, 1)} |
    | {p(12, 0)} {p(11, 0)} {p(10, 0)} {p( 9, 0)} {p( 8, 0)} {p( 7, 0)} | {p( 6, 0)} {p( 5, 0)} {p( 4, 0)} {p( 3, 0)} {p( 2, 0)} {p( 1, 0)} |
      L K J I H G   F E D C B A

    P1 jail: {"".join("X" for _ in range(int(board[25])))}
    P1 off: {"".join("X" for _ in range(int(board[26])))}
    """  # noqa

    print(out)


@njit()
def make_move_container():
    return list([empty_move()])


@njit()
def add_move_to_container(container, move):
    container.append(move)


@njit()
def empty_move():
    return np.zeros(29, dtype=np.float32)


@njit()
def convert_to_move(a, b, capture=False, ):
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
def valid_startpoint(board, x):
    if board[x] > 0:
        return True
    return False


@njit()
def captured_endpoint(board, x):
    return board[x] == -1


@njit()
def moves_for_one_die(board, die, make_unique=False):

    moves = make_move_container()

    is_game_over = game_over(board)
    if is_game_over:
        move_array = manual_concat(moves, make_unique=make_unique)
        # move_array = filter_moves_fast(move_array)
        return move_array

    # if you're in jail
    if board[25] > 0:
        a = 25
        b = a - die
        is_valid = valid_endpoint(board, b)
        is_captured = captured_endpoint(board, b)
        if is_valid:
            move = convert_to_move(a, b, capture=is_captured)
            add_move_to_container(moves, move)

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
                add_move_to_container(moves, move)

            # if all pieces are past the die value
            elif die > max_position:
                a = max_position
                b = 26
                move = convert_to_move(a, b)
                add_move_to_container(moves, move)

        # add all regular moves
        current_positions = np.where(board[0:25] > 0)[0]
        for a in current_positions:
            b = a - die
            is_valid = valid_endpoint(board, b)
            is_captured = captured_endpoint(board, b)
            if is_valid:
                move = convert_to_move(a, b, capture=is_captured)
                add_move_to_container(moves, move)

    move_array = manual_concat(moves, make_unique=make_unique)
    # move_array = filter_moves_fast(move_array)
    return move_array


@njit()
def moves_for_two_dice(board, d1, d2, all_doubles=False, make_unique=False):

    moves = make_move_container()

    # if fast=True, skip the full evaluation of all possible
    # moves with doubled dice (this will be compensated for in
    # the play_game function by running the current function 2x)
    doubles = False if not all_doubles else d1 == d2

    if doubles:

        # this part is a bit slow
        first_moves = moves_for_one_die(board, d1)
        for m1 in first_moves:
            tmp_board = board + m1
            second_moves = moves_for_one_die(tmp_board, d1)
            for m2 in second_moves:
                tmp_board = board + m1 + m2
                third_moves = moves_for_one_die(tmp_board, d1)
                for m3 in third_moves:
                    tmp_board = board + m1 + m2 + m3
                    fourth_moves = moves_for_one_die(tmp_board, d1)
                    for m4 in fourth_moves:
                        move = m1 + m2 + m3 + m4
                        add_move_to_container(moves, move)

    else:

        first_moves = moves_for_one_die(board, d1)
        for m1 in first_moves:
            tmp_board = board + m1
            second_moves = moves_for_one_die(tmp_board, d2)
            for m2 in second_moves:
                move = m1 + m2
                add_move_to_container(moves, move)

        # same again, but with second die first
        first_moves = moves_for_one_die(board, d2)
        for m1 in first_moves:
            tmp_board = board + m1
            second_moves = moves_for_one_die(tmp_board, d1)
            for m2 in second_moves:
                move = m1 + m2
                add_move_to_container(moves, move)

    move_array = manual_concat(moves, make_unique=make_unique)
    move_array = filter_moves_fast(move_array)
    return move_array


@njit()
def manual_concat(arrays, make_unique=False):
    # needed this function because numba doesn't implement
    # numpy functions like np.concatenate or np.unique

    n_cols = len(arrays[0])
    n_rows = len(arrays)
    new_array = np.zeros(shape=(n_rows, n_cols), dtype=np.float32)

    i = 0

    # checking for uniqueness is a bit slow
    if make_unique:
        for x in arrays:
            skip = False
            for j in range(i):
                if (x == new_array[j, :]).all():
                    skip = True
            if not skip:
                new_array[i, :] = x
                i += 1
    else:
        for x in arrays:
            new_array[i, :] = x
            i += 1

    return new_array


def filter_moves(move_array):
    # have to use all your moves if you're able to, so need
    # to filter out moves that don't use the maximum number
    # of dice
    max_dice_used = move_array[:, 28].max()
    move_array = move_array[move_array[:, 28] == max_dice_used, :]
    # move_array[:, 28] = 0   # Should this go here? Or in flip_board
    return move_array


filter_moves_fast = njit(filter_moves)


@njit()
def is_race(board):
    p1_pos = np.where(board[0:26] > 0)[0]
    p2_pos = np.where(board[0:26] < 0)[0]
    return p1_pos[-1] < p2_pos[0]


@njit()
def is_race_array(board_array):
    return np.array([is_race(b) for b in board_array], dtype=np.float32)


POSITIONS = np.array(range(1, 25), dtype=np.float32)


@njit()
def board_array_to_state_array(board_array):

    state_array = np.zeros(shape=(len(board_array), 79), dtype=np.float32)

    # for convenience, aliases for 'main board' and
    # a vector of main board positions
    mb = board_array[:, 1:25]
    pos = POSITIONS

    # 0-27 = actual number of piece in each position, including off-board and jail
    state_array[:, 0:28] = board_array[:, 0:28]
    # 28-51 = whether there is exactly one piece (-1, 0, 1)
    state_array[:, 28:52] = mb * (np.abs(mb) == 1)
    # 52-75 = whether there are exactly two pieces (-1, 0, 1)
    state_array[:, 52:76] = mb * (np.abs(mb) == 2)
    # 76 = pipcount for player 1
    state_array[:, 76] = (mb * pos * (mb > 0)).sum(axis=1)
    # 77 = pipcount for player 2
    state_array[:, 77] = (mb * pos * (mb < 0)).sum(axis=1)
    # 78 = whether we're in a race position
    state_array[:, 78] = is_race_array(board_array)

    return state_array


@njit()
def board_to_state(board):
    state_array = board_array_to_state_array(board.reshape(1, -1))
    return state_array.flatten()


class RandomPlayer(object):
    def action(self, board, board_array, **kwargs):
        return np.random.randint(len(board_array))


random_player = RandomPlayer()


def play_game(player1=None, player2=None, train=False, fast=True, **kwargs):
    # passing fast=True makes the game functions slightly less
    # correct, but 10x faster, so definitely will want it on
    # for model training

    if player1 is None:
        player1 = random_player
    if player2 is None:
        player2 = random_player

    board = STARTING_BOARD.copy()
    player = np.random.choice([1, -1])  # who goes first?
    turn = 0

    while True:

        dice = roll_dice()

        # if the dice are doubles, then you can make four moves
        # if fast=True, then we'll evaluate this as two consecutive
        # turns with two moves each; this is faster but may give
        # slightly different results in some edge cases
        doubles = dice[0] == dice[1] if fast else False
        all_doubles = False if fast else True

        for _ in range(int(doubles) + 1):

            move_array = moves_for_two_dice(
                board, dice[0], dice[1], all_doubles=all_doubles, make_unique=False
            )
            board_array = board + move_array

            if player == 1:
                action = player1.action(board, board_array, train=train, **kwargs)
            else:
                action = player2.action(board, board_array, train=train, **kwargs)

            board = board_array[action].copy()
            board = reset_move_counter(board)

            # check exit conditions
            is_game_over = game_over(board)
            if is_game_over:
                winner = player
                # always return board from player 1's POV
                final_board = board if player == 1 else flip_board(board)
                return winner, final_board
            if not fast:
                errors = check_for_error(board)
                if errors:
                    print(errors)
                    raise ValueError("Game in error state")

        # prep for next turn
        player *= -1
        board = flip_board(board)
        turn += 1


# def plot_perf(performance):
#     plt.plot(performance)
#     plt.show()
#     return


def main():

    n_games = 500
    n_epochs = 1_000

    # player1 = pubeval2.PubEval()
    player1 = teddy.AgentTeddy(saved_model="38_thirtyeight", model_name="~")
    player2 = pubeval2.PubEval()

    start_time = time.time()

    winners = {}
    winners[1] = 0
    winners[-1] = 0

    for g in tqdm(range(n_games)):
        if g % n_epochs == 0:
            print(winners)
        winner, _ = play_game(player1, player2, fast=True, train=False, rollout=False)
        winners[winner] += 1

    print("Winners:", winners)
    print("Player 1 win rate:", winners[1] / sum(winners.values()))

    run_time = time.time() - start_time
    print("Run time:", run_time)
    print("Average time:", run_time / n_games)

    # plot_perf(performance)


if __name__ == "__main__":

    # import cProfile
    # cProfile.run("main()")
    main()
