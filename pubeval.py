# Calling Tesauro's pubeval from Python
# To compile:
# gcc -c -fPIC pubeval.c
# gcc -shared -o libpubeval.so pubeval.o

import ctypes
import numpy as np
import backgammon
import flipped_agent
from ctypes import cdll, c_float, c_int

lib = cdll.LoadLibrary("./libpubeval.so")
lib.pubeval.restype = c_float
intp = ctypes.POINTER(ctypes.c_int)

# see if players are out of contact, then we have a pure race
def israce(board):
    p1pos = np.where(board[1:25] > 0)[0]
    p2pos = np.where(board[1:25] < 0)[0]
    if ((len(p2pos) == 0) | (len(p1pos) == 0)):
        return 1
    if (p1pos[len(p1pos)-1] < p2pos[0]):
        return 1
    return 0

# convert to Tesauro's layout
def pubeval_flip(board):
    board[[0,26]] = board[[26,27]]
    board[27] = board[28]
    board = board[:-1]
    return board

def action(board, dice, oplayer, nRoll = 0, **kwargs):
    flipped_player = -1
    if (flipped_player == oplayer):
        board = flipped_agent.flip_board(np.copy(board))
        player = -flipped_player
    else:
        player = oplayer
    # check out the legal moves available for the throw
    race = c_int(israce(board))


    # possible_moves, possible_boards = backgammon.legal_moves(board, dice, player)

    # check out the legal moves available for the throw
    possible_moves = backgammon.legal_moves(board, dice, player=1)
    possible_boards = list()
    for m in possible_moves:
        if len(m) == 1:
            new_board = backgammon.update_board(board, m[0], player=1)
        elif len(m) == 2:
            tmp_board = backgammon.update_board(board, m[0], player=1)
            new_board = backgammon.update_board(tmp_board, m[1], player=1)
        possible_boards.append(new_board)

    na = len(possible_moves)
    va = np.zeros(na)
    if (na == 0):
        return []
    for i in range(0, na):
        board = pubeval_flip(possible_boards[i])
        board = board.astype(dtype = ctypes.c_int)
        va[i] = lib.pubeval(race, board.ctypes.data_as(intp))
    action = possible_moves[np.argmax(va)]
    if (flipped_player == oplayer): # map this move to right view
        action = flipped_agent.flip_move(action)
    return action
