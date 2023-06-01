import numpy as np
import cv2
from math import inf, ceil
import json
import os

from generator_utils import compositeAdj, getSliceBounds
from char import CharSet


# Resizes targetImg to be a multiple of character width
# Scales height to correct for change in proportion
# Pads height to be a multiple of character height
def resizeTarget(im, rowLength, charShape, charChange):
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
    return im, newHeight - outHeight, rowLength


# Returns a mockup image, with the same size as the target image
def genMockup(
    comboGrid, generator, targetShape, targetPadding, crop=True, addFixed=True
):
    mockupCopy = generator.mockupImg.copy()
    gridCopy = generator.comboGrid.grid.copy()
    generator.comboGrid.grid = comboGrid
    # print(generator.comboGrid)
    for row in range(generator.rows - 1):
        for col in range(generator.cols - 1):
            startX, startY, endX, endY = getSliceBounds(
                generator, row, col, shrunken=False
            )
            generator.mockupImg[startY:endY, startX:endX] = compositeAdj(
                generator, row, col, addFixed=addFixed
            )
    # Save the new mockup, then put everything in generator back to normal
    mockup = generator.mockupImg.copy()
    generator.mockupImg = mockupCopy
    generator.comboGrid.grid = gridCopy
    # Crop and resize mockup to match target image
    if targetPadding > 0 and crop:
        mockup = mockup[:-targetPadding, :]
        # print("cropped to", mockup.shape)
    # return mockup
    resized = cv2.resize(
        mockup, dsize=(targetShape[1], targetShape[0]), interpolation=cv2.INTER_AREA
    )
    # print("mockup has shape", resized.shape)
    return resized if crop else mockup


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
