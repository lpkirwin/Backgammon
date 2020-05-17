#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backgammon interface
Run this program to play a game of Backgammon
The agent is stored in another file 
Most (if not all) of your agent-develeping code should be written in the agent.py file
Feel free to change this file as you wish but you will only submit your agent 
so make sure your changes here won't affect his performance.
"""
import numpy as np
import matplotlib.pyplot as plt
import random_agent

# import sys
import time

# import pubeval
# import kotra

# import functools
from tqdm import tqdm
from numba import njit
from numba.typed import List


def init_board():
    # initializes the game board
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
    return board[27] == 15 or board[28] == -15


@njit()
def check_for_error(board):

    error_in_game = False

    player_1_pieces = board[board > 0].sum()
    player_2_pieces = board[board < 0].sum()
    if (player_1_pieces != 15) or (player_2_pieces != -15):
        errorInProgram = True
        print("Too many or too few pieces on board!")

    return error_in_game


def pretty_print(board):
    string = str(
        np.array2string(board[1:13])
        + "\n"
        + np.array2string(board[24:12:-1])
        + "\n"
        + np.array2string(board[25:29])
    )
    print("board: \n", string)


@njit()
def legal_move(board, die, player):
    # finds legal moves for a board and one dice
    # inputs are some BG-board, the number on the die and which player is up
    # outputs all the moves (just for the one die)
    possible_moves = List()
    is_game_over = game_over(board)

    if player == 1:

        # dead piece, needs to be brought back to life
        if board[25] > 0:
            start_pip = 25 - die
            if board[start_pip] > -2:
                # possible_moves.append(np.array([25, start_pip]))
                possible_moves.append([25, start_pip])
                # possible_moves.append((25, start_pip))
                # possible_moves.append(List([25, start_pip]))

        # no dead pieces
        else:
            # adding options if player is bearing off
            is_bearing_off = (board[7:25] > 0).sum() == 0
            if is_bearing_off:
                if board[die] > 0:
                    # possible_moves.append(np.array([die, 27]))
                    possible_moves.append([die, 27])
                    # possible_moves.append((die, 27))
                    # possible_moves.append(List([die, 27]))

                elif not is_game_over:  # smá fix
                    # everybody's past the dice throw?
                    s = np.max(np.where(board[1:7] > 0)[0] + 1)
                    if s < die:
                        # possible_moves.append(np.array([s, 27]))
                        possible_moves.append([s, 27])
                        # possible_moves.append((s, 27))
                        # possible_moves.append(List([s, 27]))

            possible_start_pips = np.where(board[0:25] > 0)[0]

            # finding all other legal options
            for s in possible_start_pips:
                end_pip = s - die
                if end_pip > 0:
                    if board[end_pip] > -2:
                        # possible_moves.append(np.array([s, end_pip]))
                        possible_moves.append([s, end_pip])
                        # possible_moves.append((s, end_pip))
                        # possible_moves.append(List([s, end_pip]))

    elif player == -1:
        # dead piece, needs to be brought back to life
        if board[26] < 0:
            start_pip = die
            if board[start_pip] < 2:
                # possible_moves.append(np.array([26, start_pip]))
                possible_moves.append([26, start_pip])
                # possible_moves.append((26, start_pip))
                # possible_moves.append(List([26, start_pip]))

        # no dead pieces
        else:
            # adding options if player is bearing off
            is_bearing_off = (board[1:19] < 0).sum() == 0
            if is_bearing_off:

                if board[25 - die] < 0:
                    # possible_moves.append(np.array([25 - die, 28]))
                    possible_moves.append([25 - die, 28])
                    # possible_moves.append((25 - die, 28))
                    # possible_moves.append(List([25 - die, 28]))

                elif not is_game_over:  # smá fix
                    # everybody's past the dice throw?
                    s = np.min(np.where(board[19:25] < 0)[0])
                    if (6 - s) < die:
                        # possible_moves.append(np.array([19 + s, 28]))
                        possible_moves.append([19 + s, 28])
                        # possible_moves.append((19 + s, 28))
                        # possible_moves.append(List([19 + s, 28]))

            # finding all other legal options
            possible_start_pips = np.where(board[0:25] < 0)[0]
            for s in possible_start_pips:
                end_pip = s + die
                if end_pip < 25:
                    if board[end_pip] < 2:
                        # possible_moves.append(np.array([s, end_pip]))
                        possible_moves.append([s, end_pip])
                        # possible_moves.append((s, end_pip))
                        # possible_moves.append(List([s, end_pip]))

    return possible_moves


@njit()
def legal_moves(board, dice, player):
    # finds all possible moves and the possible board after-states
    # inputs are the BG-board, the dices rolled and which player is up
    # outputs the possible pair of moves (if they exists) and their after-states

    moves = List()
    # boards = List()

    # try using the first dice, then the second dice
    possible_first_moves = legal_move(board, dice[0], player)
    for m1 in possible_first_moves:
        temp_board = update_board(board, m1, player)
        possible_second_moves = legal_move(temp_board, dice[1], player)
        for m2 in possible_second_moves:
            # moves.append(np.array([m1, m2]))
            moves.append([m1, m2])
            # moves.append((m1, m2))
            # moves.append(List([m1, m2]))
            # boards.append(update_board(temp_board, m2, player))

    if dice[0] != dice[1]:
        # try using the second dice, then the first one
        possible_first_moves = legal_move(board, dice[1], player)
        for m1 in possible_first_moves:
            temp_board = update_board(board, m1, player)
            possible_second_moves = legal_move(temp_board, dice[0], player)
            for m2 in possible_second_moves:
                # moves.append(np.array([m1, m2]))
                moves.append([m1, m2])
                # moves.append((m1, m2))
                # moves.append(List([m1, m2]))
                # boards.append(update_board(temp_board, m2, player))

    # if there's no pair of moves available, allow one move:
    moves_exist = len(moves)
    if not moves_exist:
        # first dice:
        possible_first_moves = legal_move(board, dice[0], player)
        for m in possible_first_moves:
            # moves.append(np.array([m]))
            moves.append([m])
            # moves.append((m,))
            # moves.append(List([m]))
            # boards.append(update_board(temp_board, m, player))

        # second dice:
        if dice[0] != dice[1]:
            possible_first_moves = legal_move(board, dice[1], player)
            for m in possible_first_moves:
                moves.append([m])
                # moves.append((m,))
                # moves.append(List([m]))
                # boards.append(update_board(temp_board, m, player))

    return moves


# def is_legal_move(move, board_copy, dice, player, i):
#     if len(move) == 0:
#         return True
#     global possible_moves
#     possible_moves = legal_moves(board_copy, dice, player)
#     legit_move = np.array(
#         [np.array((possible_move == move)).all() for possible_move in possible_moves]
#     ).any()
#     if not legit_move:
#         print("Game forfeited. Player " + str(player) + " made an illegal move")
#         return False
#     return True


@njit()
def update_board(board, move, player):
    # updates the board
    # inputs are some board, one move and the player
    # outputs the updated board
    board_to_update = np.copy(board)
    move_exists = len(move)

    if not move_exists:
        return board_to_update

    startPip = move[0]
    endPip = move[1]

    # moving the dead piece if the move kills a piece
    kill = board_to_update[endPip] == (-1 * player)
    if kill:
        board_to_update[endPip] = 0
        jail = 25 + (player == 1)
        board_to_update[jail] = board_to_update[jail] - player

    board_to_update[startPip] = board_to_update[startPip] - 1 * player
    board_to_update[endPip] = board_to_update[endPip] + player

    return board_to_update


def play_a_game(player1, player2, train=False, train_config=None):
    board = STARTING_BOARD.copy()  # initialize the board
    player = np.random.randint(2) * 2 - 1  # which player begins?

    while True:

        dice = roll_dice()
        dice_match = dice[0] == dice[1]

        # make a move (2 moves if the same number appears on the dice)
        for i in range(1 + int(dice_match)):

            board_copy = np.copy(board)

            if train:
                if player == 1:
                    move = player1.action(
                        board_copy,
                        dice,
                        player,
                        i,
                        train=train,
                        train_config=train_config,
                    )
                elif player == -1:
                    move = player2.action(
                        board_copy,
                        dice,
                        player,
                        i,
                        train=train,
                        train_config=train_config,
                    )
            else:
                if player == 1:
                    move = player1.action(board_copy, dice, player, i)
                elif player == -1:
                    move = player2.action(board_copy, dice, player, i)

            # check if the move is valid
            # if not is_legal_move(move, board_copy, dice, player, i):
            #     print("Game forfeited. Player " + str(player) + " made an illegal move")
            #     return -1 * player

            # update the board
            if len(move):
                for m in move:
                    board = update_board(board, m, player)

        # players take turns
        player = -player

        is_game_over = game_over(board)
        if is_game_over:
            return -1 * player, board

        # game_has_error = check_for_error(board)
        # if game_has_error:
        #     raise ValueError("Game in error state")

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
    nGames = 500  # how many games?
    performance = list()
    player1 = random_agent
    player2 = random_agent
    wins = 0
    nEpochs = 1_000
    print("Playing " + str(nGames) + " between" + str(player1) + " and " + str(player2))
    for g in tqdm(range(nGames)):
        if g % nEpochs == 0:
            performance = log_status(g, wins, performance, nEpochs)
            wins = 0
        winner, _ = play_a_game(player1, player2)
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
