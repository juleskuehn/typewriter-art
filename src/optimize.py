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
    display=True,
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
        fractional_layer_offsets = json.load(f)[layers]

    layer_offsets = [
        (int(charHeight * o[0]), int(charWidth * o[1]))
        for o in fractional_layer_offsets
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
    (line,) = plt.plot(err_over_time)
    (point,) = plt.plot(0, 0, "ro")

    startTime = time.perf_counter_ns()
    n_comparisons = 0
    # Layer optimization passes
    for loop_num in range(num_loops):
        for layer_num, layer_offset in enumerate(layer_offsets):
            # Composite all other layers
            # bg = np.prod(np.delete(layers, layer_num, axis=0), axis=0)
            # The above code doesn't get optimized by numba. Rewrite more explicitly
            bg = layers[(layer_num + 1) % len(layer_offsets)]
            for i in range(2, len(layer_offsets)):
                bg = bg * layers[(layer_num + i) % len(layer_offsets)]

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

            iteration = loop_num * len(layer_offsets) + layer_num 
            err_over_time[iteration :] = err
            if display > 0 and (iteration % display) == 0:
                # update data
                line.set_ydata(err_over_time)
                point.set_xdata([iteration])
                point.set_ydata([err_over_time[iteration]])
                # update y axis
                plt.ylim(0, np.max(err_over_time))
                # convert it to an OpenCV image/numpy array
                fig.canvas.draw()
                # convert canvas to image
                graph_image = np.array(fig.canvas.get_renderer()._renderer)
                # it still is rgb, convert to opencv's default bgr
                graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGB2BGR)
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
            if display > 0 and (iteration % display) == 0:
                shrink = False
                mockupImg = np.array(mockup * 255, dtype="uint8")
                if max(mockupImg.shape) > 1600:
                    shrink = True
                mockupImg = cv2.cvtColor(mockupImg, cv2.COLOR_GRAY2BGR)
                if shrink:
                    mockupImg = cv2.resize(
                        mockupImg,
                        (0, 0),
                        fx=1600 / max(mockupImg.shape),
                        fy=1600 / max(mockupImg.shape),
                    )
                cv2.imshow("mockup", mockupImg)
                cv2.waitKey(1)

    endTime = time.perf_counter_ns()
    if display > 0:
        cv2.waitKey(0)

    # Save the final images
    cv2.imwrite(os.path.join(base_path, "results", "final.png"), mockup * 255)
    # Change the layer offsets back to the original values (proportions of charW / charH)
    for layer_num, layer_offset in enumerate(fractional_layer_offsets):
        cv2.imwrite(
            os.path.join(
                base_path, "results", f"layer_{layer_offset[0]}_{layer_offset[1]}.png"
            ),
            layers[layer_num] * 255,
        )
    # Save the final choices as a JSON file
    choices_dict = {}
    for layer_num, layer_offset in enumerate(fractional_layer_offsets):
        choices_dict[f"{layer_offset[0]}_{layer_offset[1]}"] = choices[
            layer_num
        ].tolist()
    with open(os.path.join(base_path, "results", "choices.json"), "w") as f:
        json.dump(choices_dict, f)

    # Save the error plot over time
    # update data
    line.set_ydata(err_over_time)
    point.set_xdata(loop_num * len(layer_offsets) + layer_num)
    point.set_ydata(err_over_time[loop_num * len(layer_offsets) + layer_num])
    # update y axis
    plt.ylim(0, np.max(err_over_time))

    # convert it to an OpenCV image/numpy array
    fig.canvas.draw()
    # convert canvas to image
    graph_image = np.array(fig.canvas.get_renderer()._renderer)
    # it still is rgb, convert to opencv's default bgr
    graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(base_path, "results", "error_over_time.png"), graph_image)

    print(f"Layer optimization took {(endTime - startTime) / 1e9} seconds")
    print(f"Total comparisons: {n_comparisons}")
    print(f"Time per comparison: {(endTime - startTime) / n_comparisons} ms")
    return


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--charset",
        "-c",
        type=str,
        default="sc-2",
        help="Path to charset folder containing config.json and image (default sc-2)",
    )
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        default="mwdog_crop.png",
        help="Path to target image in ./images (default mwdog_crop.png)",
    )
    parser.add_argument(
        "--row_length",
        "-r",
        type=int,
        default=30,
        help="Number of characters per row; determines image size (default 40)",
    )
    parser.add_argument(
        "--num_loops",
        "-n",
        type=int,
        default=15,
        help="Number of times to optimize each layer (default 20)",
    )
    parser.add_argument(
        "--init_mode",
        "-i",
        type=str,
        default="random",
        help="Start with random or blank image (default random)",
    )
    parser.add_argument(
        "--asymmetry",
        "-a",
        type=float,
        default=0.1,
        help="Asymmetry of mean squared error function (default 0.1)",
    )
    parser.add_argument(
        "--search",
        "-s",
        type=str,
        default="simAnneal",
        help="Search algorithm. Options: 'simAnneal', 'greedy' (default simAnneal)",
    )
    parser.add_argument(
        "--init_temp",
        "-temp",
        type=float,
        default=0.001,
        help="Initial temperature for simulated annealing (default 0.001)",
    )
    parser.add_argument(
        "--layers",
        "-l",
        type=str,
        default="16x1",
        help="Key to layers.json for offsets - how many layers, where to place them (default 16x1)",
    )
    parser.add_argument(
        "--display",
        "-d",
        type=int,
        default=1,
        help="Display the mockup every X iterations (1 == most often) or 0 to not display (default 1)",
    )

    args = parser.parse_args()
    kword(**vars(args))


if __name__ == "__main__":
    main()
