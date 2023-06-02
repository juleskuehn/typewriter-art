import argparse
import json
import os
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils import layer_optimization_pass, prep_charset, resizeTarget


def kword(
    charset="hermes",
    target="mwdog_crop.png",
    layers="4x1",
    row_length=20,
    num_loops=5,
    init_mode="blank",
    asymmetry=0.1,
    search="simAnneal",
    init_temp=0.001,
):
    base_path = os.getcwd()
    chars, xChange, yChange = prep_charset(charset, base_path)

    targetImg = cv2.imread(f"{base_path}/images/{target}", cv2.IMREAD_GRAYSCALE)

    charHeight, charWidth = chars[0].shape
    newTarget, targetPadding = resizeTarget(
        targetImg,
        row_length,
        (charHeight, charWidth),
        (xChange, yChange),
    )

    num_rows = newTarget.shape[0] // charHeight
    num_cols = newTarget.shape[1] // charWidth
    
    # Load layer offsets from layers.json
    with open(os.path.join(base_path, "layers.json"), "r") as f:
        layer_offsets = json.load(f)[layers]

    layer_offsets = [(int(charHeight * o[0]), int(charWidth * o[1])) for o in layer_offsets]

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

    if init_mode == "random":
        choices = np.random.randint(0, chars.shape[0], choices.shape)
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
    err_over_time = np.zeros(num_loops * len(layer_offsets))
    fig = plt.figure()
    line, = plt.plot(err_over_time)

    startTime = time.perf_counter_ns()
    n_comparisons = 0
    # Layer optimization passes
    for loop_num in range(num_loops):
        for layer_num, layer_offset in enumerate(layer_offsets):
            # Composite all other layers
            bg = np.prod(np.delete(layers, layer_num, axis=0), axis=0)

            choices[layer_num], mockup, comparisons, err = layer_optimization_pass(
                bg,
                mockup,
                target,
                chars,
                choices[layer_num],
                layer_offset,
                asymmetry=asymmetry,
                mode=search,
                temperature=init_temp / (loop_num + 1),
            )

            n_comparisons += comparisons

            err_over_time[loop_num * len(layer_offsets) + layer_num:] = err
            # update data
            line.set_ydata(err_over_time)
            # update y axis
            plt.ylim(0, np.max(err_over_time))
            # convert it to an OpenCV image/numpy array
            fig.canvas.draw()
            # convert canvas to image
            graph_image = np.array(fig.canvas.get_renderer()._renderer)
            # it still is rgb, convert to opencv's default bgr
            graph_image = cv2.cvtColor(graph_image,cv2.COLOR_RGB2BGR)
            cv2.imshow("plot", graph_image)

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

            # Show the mockup
            mockupImg = np.array(
                mockup * 255, dtype="uint8"
            )
            mockupImg = cv2.cvtColor(mockupImg, cv2.COLOR_GRAY2BGR)
            cv2.imshow("mockup", mockupImg)
            cv2.waitKey(10)

    endTime = time.perf_counter_ns()
    cv2.waitKey(0)
    print(f"Layer optimization took {(endTime - startTime) / 1e9} seconds")
    print(f"Total comparisons: {n_comparisons}")
    print(f"Time per comparison: {(endTime - startTime) / n_comparisons} ms")
    return


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--charset",
        type=str,
        default="sc-2",
        help="Path to charset folder containing config.json and image",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="mwdog_crop.png",
        help="Path to target image",
    )
    parser.add_argument(
        "--row_length",
        type=int,
        default=20,
        help="Number of characters per row (determines image size)",
    )
    parser.add_argument(
        "--num_loops",
        type=int,
        default=50,
        help="Number of times to optimize each layer",
    )
    parser.add_argument(
        "--init_mode",
        type=str,
        default="random",
        help="Set as 'random' to start optimization with random characters",
    )
    parser.add_argument(
        "--asymmetry",
        type=float,
        default=0.1,
        help="Asymmetry of mean squared error function",
    )
    parser.add_argument(
        "--search",
        type=str,
        default="simAnneal",
        help="Search algorithm. Options: 'simAnneal', 'greedy'",
    )
    parser.add_argument(
        "--init_temp",
        type=float,
        default=0.001,
        help="Initial temperature for simulated annealing",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="4x1",
        help="Key to layers.json for offsets (how many layers, where)",
    )

    args = parser.parse_args()
    kword(**vars(args))


if __name__ == "__main__":
    main()
