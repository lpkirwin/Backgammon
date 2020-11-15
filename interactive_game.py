import backgammon2 as bkg
import pubeval2
import teddy

import time

board = bkg.STARTING_BOARD

player = -1

# get_dice = bkg.roll_dice
get_dice = bkg.input_dice

opp = teddy.AgentTeddy(saved_model="cottage_current_best copy 2", name="interactive")
print(opp.config)
# opp = pubeval2.PubEval()

print()
print()
player = int(input("who goes first? player: "))

while True:

    dice = get_dice()

    if player == -1:
        board = bkg.input_move(board, player, dice)

    else:
        bkg.print_board(board, player)
        bkg.print_player(player)
        bkg.print_dice(dice)
        print(f"{opp} is planning their move...")
        time.sleep(1)

        move_array = bkg.moves_for_two_dice(
            board, dice[0], dice[1], all_doubles=True, make_unique=True
        )
        board_array = board + move_array
        action = opp.action(board, board_array, print_value=True)
        move = move_array[action]
        board = board_array[action].copy()

        print("Chosen move:")
        bkg.print_move(move, player)
        print("Done.")
        print("New board:")
        bkg.print_board(board, player)
        time.sleep(0.5)

    is_game_over = bkg.game_over(board)
    if is_game_over:
        print()
        print("----------------------------")
        print("Game over!!")
        print(f"Winner: {'player' if player == 1 else str(opp)}")
        print()
        bkg.print_board(board, player)
        print()
        break

    player *= -1
    board = bkg.flip_board(board)
    print()
    print(f"-------- Switch to player {player} --------")
    print()
