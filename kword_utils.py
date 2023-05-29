import numpy as np
import cv2
from math import inf, ceil

from generator_utils import compositeAdj, getSliceBounds


# Resizes targetImg to be a multiple of character width
# Scales height to correct for change in proportion
# Pads height to be a multiple of character height
def resizeTarget(im, rowLength, charShape, charChange, numLayers=1, maxChars=None):
    while True:
        charHeight, charWidth = charShape
        xChange, yChange = charChange
        inHeight, inWidth = im.shape
        outWidth = rowLength * charWidth
        outHeight = round((outWidth / inWidth) * inHeight * (xChange / yChange))
        if maxChars != None:
            # Ensure a maxChars will not be exceeded
            numRows = outHeight // charHeight + 1
            numChars = numRows * rowLength * 4 * numLayers
            if numChars > maxChars:
                rowLength -= 1
                outWidth = rowLength * charWidth
                outHeight = round((outWidth / inWidth) * inHeight * (xChange / yChange))
            else:
                # print("numChars:", numChars)
                break
        else:
            break

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


def brightenTarget(im, blackLevel):
    imBlack = 0  # Could test target image for this, or just leave as 0
    diff = blackLevel - imBlack
    return np.array(im * ((255 - diff) / 255) + diff, dtype="uint8")


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


# Returns ([cropped images], [padded images], (cropPosX, cropPosY))
# Cropped images are used for comparison (selection)
# Padded images can be used for reconstruction (mockup) but are not strictly necessary
def chop_charset(
    fn="hermes.png",
    numX=79,
    numY=7,
    startX=0,
    startY=0,
    xPad=0,
    yPad=0,
    shrinkX=1,
    shrinkY=1,
    blankSpace=True,
    whiteThreshold=0.95,
    basePath="",
    excludeChars=[],
):
    """
    The trick is that each quadrant needs to be integer-sized (unlikely this will occur naturally), while maintaining proportionality. So we do some resizing and keep track of the changes in proportion:

    1. Rotate/crop scanned image to align with character boundaries
    - . (period) character used for this purpose, bordering the charset:
    .........
    . A B C .
    . D E F .
    .........
    - image is cropped so that each surrounding period is cut in half
    This is done manually, before running this script.

    2. Resize UP cropped image (keeping proportion) to be evenly divisible by (number of characters in the X dimension * 2). The * 2 is because we will be chopping the characters into quadrants.
    3. Resize in the y dimension (losing proportion) to be evenly divisible by (number of characters in the Y dimension * 2). Save (resizedY / originalY) as propChange.

    4. Now the charset image can be evenly sliced into quadrants. The target image (ie. photograph of a face, etc) must be resized in the Y dimension by propChange before processing. Then, the output from processing (ie. the typed mockup) must also be resized by 1/propChange.

    The issue of characters overextending their bounds cannot be fully addressed without substantial complication. We can pad the images during chopping, and then find a crop window (character size before padding) that maintains the most information from the padded images, ie. the sum of the cropped information is lowest across the character set.
    """
    im = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    # im = imread(fn)[:,:,0]*255
    # print("charset has shape", im.shape)
    # numX = 80  # Number of columns
    # numY = 8  # Number of rows

    stepX = im.shape[1] / numX  # Slice width
    stepY = im.shape[0] / numY  # Slice height

    # Need to resize charset such that stepX and stepY are each multiples of 2
    # After this, we can shrink without loss of proportion
    # shrinkFactor = 2
    newStepX = ceil(stepX / shrinkX)
    newStepY = ceil(stepY / shrinkY)

    if newStepX % 2 != 0:
        newStepX += 1
    if newStepY % 2 != 0:
        newStepY += 1
    newStepX *= shrinkX
    newStepY *= shrinkY

    xChange = stepX / newStepX
    yChange = stepY / newStepY

    im = cv2.resize(
        im, dsize=(newStepX * numX, newStepY * numY), interpolation=cv2.INTER_AREA
    )
    # print("max char level:", np.max(im), "min char level:", np.min(im))
    # print("Actual character size", stepX, stepY)
    # print("Resized char size", newStepX, newStepY)
    # print("resized charset has shape", im.shape)

    # These need manual tuning per charset image
    startX = int(startX * newStepX)  # Crop left px
    startY = int(startY * newStepY)  # Crop top px

    tiles = []
    padded = []

    for y in range(newStepY, im.shape[0], newStepY):
        for x in range(newStepX, im.shape[1], newStepX):
            if np.sum(im[y : y + newStepY, x : x + newStepX][:, :]) < (
                newStepX * newStepY * whiteThreshold * 255
            ):
                tiles.append(
                    im[y - yPad : y + newStepY + yPad, x - xPad : x + newStepX + xPad][
                        :, :
                    ]
                )
    # Append blank tile
    if blankSpace:
        tiles.insert(
            0, np.full((newStepY + yPad * 2, newStepX + xPad * 2), 255.0, dtype="uint8")
        )

    # PADDED CHARS
    xPad = 10
    yPad = 10
    for y in range(newStepY, im.shape[0], newStepY):
        for x in range(newStepX, im.shape[1], newStepX):
            if np.sum(im[y : y + newStepY, x : x + newStepX][:, :]) < (
                newStepX * newStepY * whiteThreshold * 255
            ):
                # print('x and y:', y-yPad, x-xPad)
                padded.append(
                    im[y - yPad : y + newStepY + yPad, x - xPad : x + newStepX + xPad][
                        :, :
                    ]
                )
    # Append blank tile
    if blankSpace:
        padded.insert(
            0, np.full((newStepY + yPad * 2, newStepX + xPad * 2), 255.0, dtype="uint8")
        )
    xPad = 0
    yPad = 0

    # print(len(tiles), 'characters chopped.')
    # print(len(padded), 'padded chars.')

    a = np.array(tiles)
    #####
    # THIS PART ISN"T USED ???
    maxCroppedOut = -inf
    maxCropXY = (0, 0)  # Top left corner of crop window

    ySize, xSize = a[0].shape
    ySize -= yPad * 2  # Target crop
    xSize -= xPad * 2  # Target crop

    # Try all the crops and find the best one (the one with most white)
    for y in range(yPad * 2):
        for x in range(xPad * 2):
            croppedOut = np.sum(a) - np.sum(a[:, y : y + ySize, x : x + xSize])
            if croppedOut > maxCroppedOut:
                maxCroppedOut = croppedOut
                maxCropXY = (x, y)

    x, y = maxCropXY
    # print('cropped at ', x, y)
    # np.save('cropped_

    # PADDED CHARS
    filteredChars = [char for i, char in enumerate(padded) if i + 1 not in excludeChars]
    import os

    d = f"{basePath}results/chars/"
    filesToRemove = [os.path.join(d, f) for f in os.listdir(d)]
    for f in filesToRemove:
        os.remove(f)
    for i, char in enumerate(filteredChars):
        # print(i, char.shape)
        try:
            cv2.imwrite(f"{basePath}results/chars/{i+1}.png", char)
        except:
            continue

    return a[:, y : y + ySize, x : x + xSize], tiles, (x, y), (xChange, yChange)
