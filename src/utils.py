import json
import os
from math import ceil

import cv2
import numpy as np
from numba import njit, prange


def resizeTarget(im, rowLength, charShape, charChange):
    """
    Resizes targetImg to be a multiple of character width
    Scales height to correct for change in proportion
    Pads height to be a multiple of character height
    """
    charHeight, charWidth = charShape
    xChange, yChange = charChange
    inHeight, inWidth = im.shape
    outWidth = rowLength * charWidth
    outHeight = round((outWidth / inWidth) * inHeight * (xChange / yChange))

    im = cv2.resize(im, dsize=(outWidth, outHeight), interpolation=cv2.INTER_AREA)
    # print("resized target has shape", im.shape)
    # Pad outHeight so that it aligns with a character boundary
    if outHeight % charHeight != 0:
        newHeight = (outHeight // charHeight + 1) * charHeight
        blank = np.full((newHeight, outWidth), 255, dtype="uint8")
        blank[0:outHeight] = im
        # Extend the target image to full height by stretching the last line
        # blank[outHeight:] = np.tile(im[-1],(newHeight-outHeight,1))
        im = blank
        # print("target padded to", im.shape)
    else:
        newHeight = inHeight

    # Pad target so it has a border of 1/2 character width/height on every side
    im = np.pad(
        im,
        ((charHeight // 2, charHeight // 2), (charWidth // 2, charWidth // 2)),
        "constant",
        constant_values=255,
    )
    # If the bottom charHeight rows are fully white, remove them
    if np.all(im[-charHeight:, :] == 255):
        im = im[:-charHeight, :]
        newHeight -= charHeight
    
    # Pixels of padding added to each side of the target image
    padding = {
        'top': charHeight // 2,
        'right': charWidth // 2,
        'bottom': newHeight - outHeight + (charHeight // 2),
        'left': charWidth // 2,
    }

    return im, padding


def chop_charset(
    image_path="hermes.png",
    slicesX=79,
    slicesY=7,
    blankSpace=True,
    whiteThreshold=0.95,
    basePath="",
    excludeChars=[],
    **kwargs,
):
    """
    Crops character images from a charset image
    """
    im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    stepX = im.shape[1] / slicesX  # Slice width
    stepY = im.shape[0] / slicesY  # Slice height

    newStepX = ceil(stepX)
    newStepY = ceil(stepY)

    # Useful to have even sizes for the usual offset of half characters
    if newStepX % 2 != 0:
        newStepX += 1
    if newStepY % 2 != 0:
        newStepY += 1

    xChange = stepX / newStepX
    yChange = stepY / newStepY

    im = (
        np.array(
            cv2.resize(
                im,
                dsize=(newStepX * slicesX, newStepY * slicesY),
                interpolation=cv2.INTER_AREA,
            ),
            dtype="float32",
        )
        / 255
    )

    # Chop character image
    chars = []
    for y in range(slicesY):
        for x in range(slicesX):
            startX = int(x * newStepX)
            startY = int(y * newStepY)
            endX = int((x + 1) * newStepX)
            endY = int((y + 1) * newStepY)
            char = im[startY:endY, startX:endX]
            if np.mean(char) < whiteThreshold:
                chars.append(char)
    # Prepend blank tile
    if blankSpace:
        chars.insert(0, np.full((newStepY, newStepX), 1.0, dtype="float32"))

    chars = [char for i, char in enumerate(chars) if i + 1 not in excludeChars]

    d = os.path.join(basePath, "results", "chars")
    filesToRemove = [os.path.join(d, f) for f in os.listdir(d)]
    for f in filesToRemove:
        os.remove(f)
    for i, char in enumerate(chars):
        try:
            cv2.imwrite(os.path.join(d, f"{i+1}.png"), char * 255)
        except:
            continue

    return np.array(chars, dtype="float32"), xChange, yChange


def prep_charset(config_dir, base_path=""):
    """
    Loads charset config file & image and prepares charset for use
    Inputs: Path to config directory (or config dict + path to charset image)
    """
    config_dir = os.path.join(base_path, "charsets", config_dir)

    with open(os.path.join(config_dir, "config.json"), "r") as f:
        config_dict = json.load(f)

    config_dict["image_path"] = os.path.join(config_dir, config_dict["image_path"])

    chars, xChange, yChange = chop_charset(
        basePath=base_path,
        **config_dict,
    )
    return chars, xChange, yChange


@njit(parallel=True, fastmath=True)
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
    total_err = 0

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
        total_err += cur_amse

    return choices, mockup, np.sum(comparisons), total_err / choices.shape[0]
