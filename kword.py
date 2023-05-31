import numpy as np
import cv2
import json
import time
import os

# njit
from numba import njit, prange

from kword_utils import resizeTarget, genMockup, prep_charset


def kword(
    charset_path="hermes",
    target_fn="mwdog_crop.png",
    rowLength=20,
    num_loops=5,
    initMode="blank",
    asymmetry=0.1,
    search="simAnneal",
    initTemp=0.001,
):
    base_path = os.getcwd()
    chars, xChange, yChange = prep_charset(charset_path, base_path)

    targetImg = cv2.imread(f"{base_path}/images/{target_fn}", cv2.IMREAD_GRAYSCALE)

    charHeight, charWidth = chars[0].shape
    resizedTarget, targetPadding, rowLength = resizeTarget(
        targetImg,
        rowLength,
        (charHeight, charWidth),
        (xChange, yChange),
    )

    # # 2023 rewrite: pad target so it has a border of 1/2 character width/height on every side
    newTarget = np.pad(
        resizedTarget,
        ((charHeight // 2, charHeight // 2), (charWidth // 2, charWidth // 2)),
        "constant",
        constant_values=255,
    )
    # If the bottom charHeight rows are fully white, remove them
    if np.all(newTarget[-charHeight:, :] == 255):
        newTarget = newTarget[:-charHeight, :]

    num_rows = newTarget.shape[0] // charHeight
    num_cols = newTarget.shape[1] // charWidth
    layer_offsets = [
        (0, 0),
        (0, charWidth // 2),
        (charHeight // 2, 0),
        (charHeight // 2, charWidth // 2),
        (0, 0),
        (0, charWidth // 2),
        (charHeight // 2, 0),
        (charHeight // 2, charWidth // 2),
    ]

    # Internal representations of images should be float32 for easy math
    target = newTarget.astype("float32") / 255
    assert target.shape[0] % chars.shape[1] == 0
    assert target.shape[1] % chars.shape[2] == 0
    mockup = np.full(target.shape, 1, dtype="float32")

    # Pad mockup and target by the maximum layer_offset in each dimension
    mockup, target = (
        np.pad(
            img,
            (
                (0, max([o[0] for o in layer_offsets])),
                (0, max([o[1] for o in layer_offsets])),
            ),
            "constant",
            constant_values=1,
        )
        for img in (mockup, target)
    )

    # Layers is a 3D array of mockups, one per layer
    layers = np.array([mockup.copy() for _ in layer_offsets], dtype="float32")

    # Choices is a 3D array of indices for chars selected per layer
    choices = np.zeros((layers.shape[0], num_rows * num_cols), dtype="uint16")

    if initMode == "random":
        choices = np.random.randint(0, len(charSet.chars), choices.shape)
        for layer_num, layer_offset in enumerate(layer_offsets):
            for i, choice in enumerate(choices[layer_num]):
                row = i // num_cols
                col = i % num_cols
                layers[layer_num][
                    row * chars.shape[1]
                    + layer_offset[0] : (row + 1) * chars.shape[1]
                    + layer_offset[0],
                    col * chars.shape[2]
                    + layer_offset[1] : (col + 1) * chars.shape[2]
                    + layer_offset[1],
                ] = chars[choice]
        mockup = np.prod(layers, axis=0)

    # Padded chars are only used for user-facing mockups - they can stay as uint8
    # padded_chars = np.array([c.padded for c in charSet.chars], dtype="uint8")

    startTime = time.perf_counter_ns()
    n_comparisons = 0
    # Layer optimization passes
    for loop_num in range(num_loops):
        for layer_num, layer_offset in enumerate(layer_offsets):
            # Composite all other layers
            bg = np.prod(np.delete(layers, layer_num, axis=0), axis=0)

            choices[layer_num], mockup, comparisons = layer_optimization_pass(
                bg,
                mockup,
                target,
                chars,
                choices[layer_num],
                layer_offset,
                asymmetry=asymmetry,
                mode=search,
                temperature=initTemp / (loop_num + 1),
            )

            n_comparisons += comparisons

            # Update the layer image based on the returned choices
            for i, choice in enumerate(choices[layer_num]):
                row = i // num_cols
                col = i % num_cols
                layers[layer_num][
                    row * chars.shape[1]
                    + layer_offset[0] : (row + 1) * chars.shape[1]
                    + layer_offset[0],
                    col * chars.shape[2]
                    + layer_offset[1] : (col + 1) * chars.shape[2]
                    + layer_offset[1],
                ] = chars[choice]

            # Write the mockup image to disk
            mockupImg = np.array(
                mockup * 255, dtype="uint8"
            )  # convert to uint8 for display
            mockupImg = cv2.cvtColor(mockupImg, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(
                os.path.join(
                    base_path, "results", f"mockup_{loop_num}_{layer_num}.png"
                ),
                mockupImg,
            )

    endTime = time.perf_counter_ns()
    print(f"Layer optimization took {(endTime - startTime) / 1e9} seconds")
    print(f"Total comparisons: {n_comparisons}")
    print(f"Time per comparison: {(endTime - startTime) / n_comparisons} ms")
    return


@njit(parallel=True, fastmath=True)
# @njit
def layer_optimization_pass(
    bg,
    mockup,
    target,
    chars,
    choices,
    layer_offset,
    asymmetry=0.1,
    mode="greedy",
    temperature=0.001,
):
    num_cols = target.shape[1] // chars.shape[2]

    comparisons = np.zeros(choices.shape[0], dtype="uint32")

    # for i, prev_choice in enumerate(choices):
    for i in prange(choices.shape[0]):
        prev_choice = choices[i]
        row = i // num_cols
        col = i % num_cols
        # Get the slices of target, mockup and background for this position
        target_slice, mockup_slice, bg_slice = [
            img[
                row * chars.shape[1]
                + layer_offset[0] : (row + 1) * chars.shape[1]
                + layer_offset[0],
                col * chars.shape[2]
                + layer_offset[1] : (col + 1) * chars.shape[2]
                + layer_offset[1],
            ]
            for img in (target, mockup, bg)
        ]
        # Get the character that was previously chosen for this position
        cur_composite = mockup_slice
        err = target_slice - cur_composite
        asym_err = np.where(err > 0, err * (1 + asymmetry), err)
        cur_amse = np.mean(np.square(asym_err))
        # Try other characters in this position
        for new_choice in np.random.permutation(chars.shape[0]):
            comparisons[i] += 1
            if new_choice == prev_choice:
                continue
            new_char = chars[new_choice]
            new_composite = bg_slice * new_char
            err = target_slice - new_composite
            asym_err = np.where(err > 0, err * (1 + asymmetry), err)
            new_amse = np.mean(np.square(asym_err))
            if mode == "greedy":
                #  Greedy: Find the best choice
                if new_amse < cur_amse:
                    choices[i] = new_choice
                    cur_amse = new_amse
                    cur_composite = new_composite
            elif mode == "simAnneal":
                # Simulated annealing: Find a (usually) better choice
                delta = cur_amse - new_amse
                if delta > 0:
                    choices[i] = new_choice
                    cur_amse = new_amse
                    cur_composite = new_composite
                    break
                try:
                    p = np.exp(delta / temperature)
                    rand_choice = p > np.random.rand()
                except:
                    rand_choice = False
                if rand_choice:
                    choices[i] = new_choice
                    cur_amse = new_amse
                    cur_composite = new_composite
                    break
        # End character comparison loop for this position
        mockup[
            row * chars.shape[1]
            + layer_offset[0] : (row + 1) * chars.shape[1]
            + layer_offset[0],
            col * chars.shape[2]
            + layer_offset[1] : (col + 1) * chars.shape[2]
            + layer_offset[1],
        ] = cur_composite

    return choices, mockup, np.sum(comparisons)


def save_results():
    return
    # Save mockup image
    mockupImg = generator.mockupImg
    if targetPadding > 0:  # Crop and resize mockup to match target image
        mockupImg = mockupImg[:-targetPadding, :]
    if hiResMockup:
        # Correct aspect
        newHeight = int(mockupImg.shape[1] * targetImg.shape[0] / targetImg.shape[1])
        resized = cv2.resize(
            mockupImg,
            dsize=(mockupImg.shape[1], newHeight),
            interpolation=cv2.INTER_AREA,
        )
    else:
        # Match to target image
        resized = cv2.resize(
            mockupImg,
            dsize=(targetImg.shape[1], targetImg.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    cv2.imwrite(basePath + "results/mockup_optimized.png", resized)

    # Save config
    with open(f"{basePath}results/config.json", "w") as f:
        f.write(json.dumps(config))

    # Save combogrid
    np.savetxt(
        f"{basePath}results/grid_optimized.txt",
        generator.comboGrid.getPrintable(),
        fmt="%i",
        delimiter=" ",
    )

    # Save layer images
    if genLayers:
        layerNames = ["BR", "BL", "TR", "TL"]
        for i, layer in enumerate(generator.comboGrid.getLayers()):
            layerImg = genMockup(
                layer,
                generator,
                targetImg.shape,
                targetPadding,
                crop=False,
                addFixed=False,
            )
            cv2.imwrite(f"{basePath}results/typeable_{layerNames[i]}.png", layerImg)

    # Save score history
    # TODO make this nices (how many positions optimized?)
    with open(f"{basePath}results/history_scores.txt", "w") as f:
        f.write("\n".join([f"{s[0]} {s[1]} {s[2]}" for s in generator.psnrHistory]))

    # Save choice history
    with open(f"{basePath}results/history_choices.txt", "w") as f:
        f.write("Row Col ChosenID\n")
        f.write(
            "\n".join(
                [
                    str(c[0]) + " " + str(c[1]) + " " + str(c[2])
                    for c in generator.choiceHistory
                ]
            )
        )

    ############################
    # Calculate scores on result, print and save
    stats = f"{generator.frame} positions optimized\n{generator.stats['comparisonsMade']} comparisons made\n{(endTime-startTime)/1000000000:.2f} seconds"

    # Calculate average time per comparison
    if generator.stats["comparisonsMade"] > 0:
        stats += f"\n{(endTime-startTime)/generator.stats['comparisonsMade']:.2f} ns per comparison"
    print(stats)

    with open(f"{basePath}results/history_stats.txt", "w", encoding="utf8") as f:
        f.write(stats)

    # Overlay the original image for comparison
    # cv2.imwrite(mockupFn+'c.png', cv2.addWeighted(resized,0.5,targetImg,0.5,0))


def main():
    kword(
        charset_path="smith_corona",
        target_fn="mwdog_crop.png",
        rowLength=20,
        num_loops=5,
        initMode="blank",
        asymmetry=0.1,
        search="simAnneal",
        initTemp=0.01,
    )


if __name__ == "__main__":
    main()
