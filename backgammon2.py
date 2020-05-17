
import time

import numpy as np
from numba import njit
from tqdm import tqdm

import pubeval2

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
def game_over_array(board):
    return (board[:, 26] == 15) | (board[:, 27] == -15)


@njit()
def check_for_error(board):
    error_in_game = False
    player_1_pieces = board[board > 0].sum()
    player_2_pieces = board[board < 0].sum()
    if (player_1_pieces != 15) or (player_2_pieces != -15):
        error_in_game = True
    return error_in_game


POSITION_NAMES = (
    ["P2 jail"]
    + list("ABCDEFGHIJKLMNOPQRSTUVWX")
    + ["P1 jail", "P1 off", "P2 off", "Blank"]
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


def print_move(move):
    for i, x in enumerate(move):
        name = POSITION_NAMES[i]
        if x > 0:
            print(f"{name}: {''.join('+' for _ in range(int(x)))}")
        if x < 0:
            print(f"{name}: {''.join('-' for _ in range(abs(int(x))))}")


def input_dice():
    dice = input("Enter dice roll: ")
    dice = make_dice(int(dice[0]), int(dice[1]))
    return dice


@njit()
def make_dice(a, b):
    # make dice that numba won't reject
    return np.array([a, b])


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

        if len(legal_moves) == 1:
            print("No legal moves?")

        move_string = input(f">> Enter move {i}: ")

        # if no input, ask again
        if not move_string:
            continue

        # convention for picking random move (possibly)
        if move_string == " ":
            if len(dice_list) != n_dice:
                print("Already started move, too late for random")
            print("Making random moves")
            for _ in range(int(doubles) + 1):
                move_array = moves_for_two_dice(board, dice)
                board_array = board + move_array
                random_player = RandomPlayer()
                action = random_player.action(board, board_array)
                chosen_move = move_array[action]
                board = board + chosen_move
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
            is_valid = any([(move == m).all() for m in legal_moves])
        except Exception:
            is_valid = False

        if is_valid:
            board = board + move
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
    """

    print(out)


@njit()
def make_move_container():
    return list([empty_move()])


@njit()
def add_move_to_container(container, move):
    container.append(move)


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
def valid_startpoint(board, x):
    if board[x] > 0:
        return True
    return False


@njit()
def captured_endpoint(board, x):
    return board[x] == -1


@njit()
def moves_for_one_die(board, die):

    moves = make_move_container()

    is_game_over = game_over(board)
    if is_game_over:
        return moves

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

    return moves


@njit()
def moves_for_two_dice(board, dice):

    # # integrity checks
    # assert board[0] <= 0
    # assert board[25] >= 0
    # assert board[26] >= 0
    # assert board[27] <= 0
    # assert board[28] == 0

    moves = make_move_container()

    # doubles = dice[0] == dice[1]
    doubles = False

    if doubles:

        # this seems to be slow :(
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


@njit()
def is_race(board):
    p1_pos = np.where(board[0:26] > 0)[0]
    p2_pos = np.where(board[0:26] < 0)[0]
    if p1_pos[-1] < p2_pos[0]:
        return True
    return False


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
        # doubles = False

        for _ in range(int(doubles) + 1):

            move_array = moves_for_two_dice(board, dice)
            board_array = board + move_array

            if player == 1:
                action = player1.action(board, board_array, train=train)
            else:
                action = player2.action(board, board_array, train=train)

            board = board_array[action].copy()

            # check exit conditions
            # game_has_error = check_for_error(board)
            # if game_has_error:
            #     raise ValueError("Game in error state")
            is_game_over = game_over(board)
            if is_game_over:
                winner = player
                return winner

        # prep for next turn
        player *= -1
        board = flip_board(board)
        turn += 1

        # if turn % 1000 == 0:
        #     print("check in")


# def plot_perf(performance):
#     plt.plot(performance)
#     plt.show()
#     return


# def log_status(g, wins, performance, nEpochs):
#     if g == 0:
#         return performance
#     print("game number", g)
#     win_rate = wins / nEpochs
#     print("win rate:", win_rate)
#     performance.append(win_rate)
#     return performance


def main():

    n_games = 5_000
    n_epochs = 1_000

    player1 = pubeval2
    player2 = pubeval2

    start_time = time.time()

    winners = {}
    winners[1] = 0
    winners[-1] = 0

    print("Playing " + str(n_games) + " between " + str(player1) + " and " + str(player2))

    for g in tqdm(range(n_games)):
        if g % n_epochs == 0:
            print(winners)
        winner = play_game(player1, player2)
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
