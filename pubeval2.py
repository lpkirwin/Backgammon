# Calling Tesauro's pubeval from Python
# To compile:
# gcc -c -fPIC pubeval.c
# gcc -shared -o libpubeval.so pubeval.o

import ctypes
from ctypes import c_float, c_int, cdll

import numpy as np
from numba import njit

import backgammon2

lib = cdll.LoadLibrary("./libpubeval.so")
lib.pubeval.restype = c_float
intp = ctypes.POINTER(ctypes.c_int)


# @njit()
def action(board, board_array, **kwargs):

    is_race = c_int(int(backgammon2.is_race(board)))

    board = board[:28]
    board_array = board_array[:, :28].astype(dtype=c_int)

    values = list()
    for new_board in board_array:
        value = lib.pubeval(is_race, new_board.ctypes.data_as(intp))
        values.append(value)

    return np.argmax(values)
