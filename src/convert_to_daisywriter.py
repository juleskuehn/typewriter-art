# This program will convert the output of the typewriter-art project to
# binary instructions for the DaisyWriter. The DW must be in DaisyPlot
# mode for these instructions to work.
#
# The format for grid_optimized.txt is:
#  * First line is all 1s (1s are spaces)
#  * Even lines are the top left and top right layers (TL) (TR) (TL) (TR) until EOL
#  * Odd lines are the bottom LR layers (BL) (BR) (BL) (BR) until EOL
#  * The numbers in each spot correspond to the character number when split (in chars dir)
#
# This program will map the char # to the raw bytes corresponding to the DaisyWriter instructions.
# Must move 1/2 of one character forward then print the next character. At EOL work backwards for
# the next line.

import math
from itertools import zip_longest
import sys

DAISYWRITER_GRAPHICS_MODE_ENTER = b"\x1B\x2E"  # ESC .
DAISYWRITER_RELATIVE_MOVEMENT_HORIZONTAL_ENTER = (
    b"\x1B\x1B\x4A\x1B\x48"  # ESC ESC J ESC H
)
DAISYWRITER_RELATIVE_MOVEMENT_VERTICAL_ENTER = (
    b"\x1B\x1B\x4A\x1B\x56"  # ESC ESC J ESC V
)
DAISYWRITER_RELATIVE_MOVEMENT_EXIT = b"\x1B\x1B\x4F"  # ESC ESC O

# All values from DaisyWriter manual page 14-9
DAISYWRITER_RELATIVE_MOVEMENT_MAX = 1791  # 1536 + 240 + 15
DAISYWRITER_RELATIVE_MOVEMENT_N1POS_VALUES = {
    1536: b"\x46",
    1280: b"\x45",
    1024: b"\x44",
    768: b"\x43",
    512: b"\x42",
    256: b"\x41",
    0: b"\x40",
}
DAISYWRITER_RELATIVE_MOVEMENT_N1NEG_VALUES = {
    1536: b"\x56",
    1280: b"\x55",
    1024: b"\x54",
    768: b"\x53",
    512: b"\x52",
    256: b"\x51",
    0: b"\x50",
}

DAISYWRITER_RELATIVE_MOVEMENT_N2_VALUES = {
    240: b"\x4F",
    224: b"\x4E",  # 225 in book, misprint?
    208: b"\x4D",  # 209 in book, misprint?
    192: b"\x4C",
    176: b"\x4B",
    160: b"\x4A",
    144: b"\x49",
    128: b"\x48",
    112: b"\x47",
    96: b"\x46",
    80: b"\x45",
    64: b"\x44",
    48: b"\x43",
    32: b"\x42",
    16: b"\x41",
    0: b"\x40",
}

DAISYWRITER_RELATIVE_MOVEMENT_N3_VALUES = {
    15: b"\x4F",
    14: b"\x4E",
    13: b"\x4D",
    12: b"\x4C",
    11: b"\x4B",
    10: b"\x4A",
    9: b"\x49",
    8: b"\x48",
    7: b"\x47",
    6: b"\x46",
    5: b"\x45",
    4: b"\x44",
    3: b"\x43",
    2: b"\x42",
    1: b"\x41",
    0: b"\x40",
}

DAISYWRITER_RELATIVE_MOVEMENT_HORIZONTAL_DEFAULT = (
    4  # 3  # 0.0825 / (1/60) = 4.95, need ~half
)

DAISYWRITER_RELATIVE_MOVEMENT_VERTICAL_DEFAULT = (
    3  # 4  # 0.1665in / (1/48) = 7.992, need ~half
)

MAX_PAPER_WIDTH_DEFAULT = 248
MAX_PAPER_HEIGHT_DEFAULT = 170


def convert_charimg_to_byte(image_number):
    print("convert_charimg_to_byte", image_number)
    assert image_number > 0
    assert image_number < 96
    return (image_number + 31).to_bytes(1, "big")


# this function must take in a value between -1791 and 1791
# and convert it to three bytes, which must follow either a
# horizontal or vertical movement command.
# the manual specifies that the command must be in the form of
# ESC ESC J ESC H/V N1 N2 N3
# N1/N2/N3 is what this will output
def convert_relative_motion(amount_to_move):
    movement_is_negative = amount_to_move < 0
    amount_to_move = abs(amount_to_move)

    assert amount_to_move < DAISYWRITER_RELATIVE_MOVEMENT_MAX

    if movement_is_negative:
        n1 = b"\x50"  # zero, but specifies reverse movement
    else:
        n1 = b"\x40"  # zero, forward movement

    n2, n3 = b"\x40", b"\x40"  # 0,0

    for key in DAISYWRITER_RELATIVE_MOVEMENT_N1POS_VALUES:
        if key == 0:  # avoid div by zero
            continue
        if amount_to_move % key != amount_to_move:
            if movement_is_negative:
                n1 = DAISYWRITER_RELATIVE_MOVEMENT_N1NEG_VALUES[key]
            else:
                n1 = DAISYWRITER_RELATIVE_MOVEMENT_N1POS_VALUES[key]

            amount_to_move -= key
            break

    for key in DAISYWRITER_RELATIVE_MOVEMENT_N2_VALUES:
        if key == 0:  # avoid div by zero
            continue
        if amount_to_move % key != amount_to_move:
            n2 = DAISYWRITER_RELATIVE_MOVEMENT_N2_VALUES[key]
            amount_to_move -= key
            break

    for key in DAISYWRITER_RELATIVE_MOVEMENT_N3_VALUES:
        if key == 0:  # avoid div by zero
            continue
        if amount_to_move % key != amount_to_move:
            n3 = DAISYWRITER_RELATIVE_MOVEMENT_N3_VALUES[key]
            amount_to_move -= key
            break
    assert amount_to_move == 0  # should be no remainder at this point
    return n1 + n2 + n3


if __name__ == "__main__":
    output_binary = b""
    centering = True

    # we read the entire grid into memory so we can calculate padding and add
    # as we loop through each row
    input_rows = []

    # put in graphics mode first
    output_binary += DAISYWRITER_GRAPHICS_MODE_ENTER

    with open("results/grid_optimized.txt", "r") as fin:
        line_number = 0
        col_len = None
        row_len = None
        move_right = True
        for line in fin:
            line_number += 1
            print("Pre-Processing line " + str(line_number))
            if line_number == 1:  # skip first line, it's always all ones
                continue

            numbers = [convert_charimg_to_byte(int(n)) for n in line.split()]
            assert numbers is not None
            if centering:
                # need to pad horizontal first, can't do vertical yet
                # since we don't know total number of rows
                col_len = len(numbers)
                assert col_len < MAX_PAPER_WIDTH_DEFAULT
                assert col_len > 0
                amount_to_pad_horizontal = math.floor(
                    (MAX_PAPER_WIDTH_DEFAULT - col_len) / 2
                )
                padding = [b" " for n in range(amount_to_pad_horizontal)]
                print(padding)
                assert isinstance(padding, list)
                numbers = padding + numbers + padding
                assert isinstance(numbers, list)
            input_rows.append(numbers)

        if centering:
            # now do the vertical centering
            assert col_len is not None
            row_len = len(input_rows)
            amount_to_pad_vertical = math.floor(
                (MAX_PAPER_HEIGHT_DEFAULT - row_len) / 2
            )
            pad_row = [b" " for n in range(col_len)]
            assert isinstance(pad_row, list)

            print("amount to pad: " + str(amount_to_pad_vertical))
            for i in range(amount_to_pad_vertical):
                print(str(i))
                input_rows.insert(0, pad_row)
                input_rows.append(pad_row)

            assert isinstance(input_rows, list)
            print("New horizontal: " + str(len(input_rows[0])))
            print("New vertical: " + str(len(input_rows)))

        line_number = 0
        print(input_rows)
        for row in input_rows:
            line_number += 1
            print("Processing line " + str(line_number))

            if line_number % 2 == 0:
                row = list(reversed(row))  # odd numbered lines will print in reverse
                print("Reverse")
                move_right = False
            else:
                print("Forward")
                move_right = True

            num_spaces = 0
            for n, lookahead in zip_longest(row, row[1:]):
                if n == b" ":
                    # writing a space
                    num_spaces += 1
                    if lookahead == b" ":
                        continue
                    output_binary += DAISYWRITER_RELATIVE_MOVEMENT_HORIZONTAL_ENTER
                    if move_right:
                        output_binary += convert_relative_motion(
                            DAISYWRITER_RELATIVE_MOVEMENT_HORIZONTAL_DEFAULT
                            * num_spaces
                        )

                    else:
                        output_binary += convert_relative_motion(
                            -DAISYWRITER_RELATIVE_MOVEMENT_HORIZONTAL_DEFAULT
                            * num_spaces
                        )

                    output_binary += DAISYWRITER_RELATIVE_MOVEMENT_EXIT
                    num_spaces = 0
                else:
                    # hack to remove !s
                    if n == b"!":
                        n = b":"
                        print("swapped ! for :")
                    print(n)
                    output_binary += n
                    output_binary += DAISYWRITER_RELATIVE_MOVEMENT_HORIZONTAL_ENTER
                    if move_right:
                        output_binary += convert_relative_motion(
                            DAISYWRITER_RELATIVE_MOVEMENT_HORIZONTAL_DEFAULT
                        )

                    else:
                        output_binary += convert_relative_motion(
                            -DAISYWRITER_RELATIVE_MOVEMENT_HORIZONTAL_DEFAULT
                            # * 3  # go back over the character (2 steps) and then one additional half character
                        )

                    output_binary += DAISYWRITER_RELATIVE_MOVEMENT_EXIT

            # after we're done move down one half of one character
            output_binary += DAISYWRITER_RELATIVE_MOVEMENT_VERTICAL_ENTER
            output_binary += convert_relative_motion(
                DAISYWRITER_RELATIVE_MOVEMENT_VERTICAL_DEFAULT
            )

            output_binary += DAISYWRITER_RELATIVE_MOVEMENT_EXIT
            if move_right:
                output_binary += DAISYWRITER_RELATIVE_MOVEMENT_HORIZONTAL_ENTER
                output_binary += convert_relative_motion(
                    -DAISYWRITER_RELATIVE_MOVEMENT_HORIZONTAL_DEFAULT
                )

                output_binary += DAISYWRITER_RELATIVE_MOVEMENT_EXIT
    output_binary += b"\r\n"  # exit graphics mode

    with open("results/output.bin", "wb", 0) as fout:
        fout.write(output_binary)
