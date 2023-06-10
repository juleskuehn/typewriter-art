import json


def convert_choices_to_positions(
    f="./results/choices.json",
    reverse_sort_every_second_row=True,
    relative_offsets=True,
    skip_zeros=True,
):
    with open(f, "r") as f:
        choices = f.read()
    # Choices is an object with keys for each layer;
    # values are a list of charIds for that layer
    choices = json.loads(choices)
    # List of tuples (charPosX, charPosY, charId)
    positions = []
    for layer_info, choice_list in choices.items():
        offset_v, offset_h = layer_info.split("_")[1:]
        for i, row in enumerate(choice_list):
            for j, col in enumerate(row):
                positions.append((float(offset_v) + i, float(offset_h) + j, col))
    positions = sorted(positions, key=lambda x: (x[0], x[1]))

    if reverse_sort_every_second_row:
        # Every second row should be reverse sorted.
        # We know that a new row has been entered when the x value changes.
        # We can use this to determine when to reverse sort.
        start_reverse_idx = -1
        for i, (x, y, char_id) in enumerate(positions):
            if i == 0:
                continue
            if x != positions[i - 1][0]:
                if start_reverse_idx != -1:
                    positions[start_reverse_idx:i] = positions[start_reverse_idx:i][
                        ::-1
                    ]
                    start_reverse_idx = -1
                else:
                    start_reverse_idx = i

        # If the last row should have been reverse sorted but wasn't, do it now.
        if start_reverse_idx != -1:
            positions[start_reverse_idx:] = positions[start_reverse_idx:][::-1]

    if skip_zeros:
        # Remove all positions with charId 0 (keeping first position always)
        positions = [p for i, p in enumerate(positions) if p[2] != 0 or i == 0]

    if relative_offsets:
        # The first position is always 0, 0.
        # Every other position should be relative to the one before it.
        relative_positions = []
        for i, (x, y, char_id) in enumerate(positions):
            if i == 0:
                relative_positions.append((x, y, char_id))
            else:
                relative_positions.append(
                    (x - positions[i - 1][0], y - positions[i - 1][1], char_id)
                )
        positions = relative_positions

    return positions


if __name__ == "__main__":
    positions = convert_choices_to_positions()
    print("offset_v offset_h char_id")
    for x, y, char_id in positions:
        print(f"{x:.3f} {y:.3f} {char_id}")
