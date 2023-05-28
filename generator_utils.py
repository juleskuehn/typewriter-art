import numpy as np
import cv2
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


def getSliceBounds(generator, row, col, shrunken=False):
    h = generator.shrunkenComboH if shrunken else generator.comboH
    w = generator.shrunkenComboW if shrunken else generator.comboW
    startY = row * h
    startX = col * w
    endY = (row + 2) * h
    endX = (col + 2) * w
    return startX, startY, endX, endY


def getSimAnneal(generator, row, col):
    # Get score of existing slice
    curScore = compare(generator, row, col)
    chars = generator.charSet.getAll()[:]
    np.random.shuffle(chars)
    newChar = None
    chars = [c for c in chars if c.id != generator.comboGrid.get(row, col)[3]]

    origGrid = generator.comboGrid.grid.copy()
    # Get composite of 3 chars underneath this one to speed up comparisons
    # Set this char to blank
    generator.comboGrid.put(row, col, 0)

    def toFloat(img):
        return np.array(img / 255, dtype="float32")

    # Gather primitives needed for parallel processing - these are common to all chars
    otherCharsComposite = toFloat(compositeAdj(generator, row, col))
    startX, startY, endX, endY = getSliceBounds(generator, row, col, shrunken=False)
    targetSlice = generator.targetImg[startY:endY, startX:endX]

    # Generator for parallel processing including all primitives
    vars_for_compare = (
        (
            targetSlice,
            otherCharsComposite,
            char.cropped,
            char.id,
            curScore,
            generator.asym,
            generator.compareMode,
        )
        for char in chars
    )

    def compositeAndCompare(
        targetSlice,
        otherCharsComposite,
        charImg,
        charID,
        curScore,
        asymmetry,
        compareMode,
    ):
        mockupSlice = charImg * otherCharsComposite
        score = 0
        if compareMode in ["ssim"]:
            score = -1 * compare_ssim(targetSlice, mockupSlice) + 1
        elif compareMode in ["amse"]:
            score = compare_amse(targetSlice, mockupSlice, asymmetry)
        elif compareMode in ["blend"]:
            ssim = -1 * compare_ssim(targetSlice, mockupSlice) + 1
            amse = compare_amse(targetSlice, mockupSlice, asymmetry)
            amse = np.sqrt(amse) / 255
            score = amse * ssim**0.5  # Hardcoded my usual blend function

        # Note that delta is reversed because we are looking for a minima
        delta = curScore - score
        return delta, charID

    for vars in vars_for_compare:
        delta, charID = compositeAndCompare(*vars)
        generator.stats["comparisonsMade"] += 1
        if delta > 0:
            newChar = charID
            break
        simRand = np.exp(delta / (generator.scaleTemp * generator.temperature))
        randChoice = simRand > np.random.rand()
        if randChoice:
            generator.randomChoices += 1
            newChar = charID
            break
    generator.comboGrid.grid = origGrid
    # generator.queue.add((row, col))
    return newChar


def compare(generator, row, col):
    asymmetry = generator.asym
    generator.stats["comparisonsMade"] += 1
    startX, startY, endX, endY = getSliceBounds(generator, row, col, shrunken=False)
    targetSlice = generator.targetImg[startY:endY, startX:endX]
    mockupSlice = compositeAdj(generator, row, col, shrunken=False)
    score = 0
    if generator.compareMode in ["ssim"]:
        score = -1 * compare_ssim(targetSlice, mockupSlice) + 1
    elif generator.compareMode in ["amse"]:
        score = compare_amse(targetSlice, mockupSlice, asymmetry)
    elif generator.compareMode in ["blend"]:
        ssim = -1 * compare_ssim(targetSlice, mockupSlice) + 1
        amse = compare_amse(targetSlice, mockupSlice, asymmetry)
        amse = np.sqrt(amse) / 255
        score = generator.blendFunc(amse, ssim)

    return score


def dirtyLinearPositions(generator, randomOrder=False, zigzag=False):
    positions = []
    layers = [0, 3, 2, 1] if zigzag else [0, 3, 1, 2]
    for layerID in layers:
        startIdx = len(positions)
        # r2l = False if np.random.rand() < 0.5 else True
        startRow = 0
        startCol = 0
        endRow = generator.rows - 1
        endCol = generator.cols - 1
        if layerID in [2, 3]:
            startRow = 1
            # r2l = True
        if layerID in [1, 3]:
            startCol = 1
        for row in range(startRow, endRow, 2):
            for col in range(startCol, endCol, 2):
                if generator.comboGrid.isDirty(row, col):
                    # if generator.comboGrid.isDirty(row,col):
                    positions.append((row, col))
                else:
                    generator.comboGrid.clean(row, col)
        if zigzag and layerID in [2, 3]:
            positions[startIdx : len(positions)] = positions[
                len(positions) - 1 : startIdx - 1 : -1
            ]
        if len(positions) > 0:
            positions.append(None)
    if randomOrder:
        np.random.shuffle(positions)
    # print(positions)
    # exit()
    return positions


def initRandomPositions(generator):
    # Initialize randomly if desired
    numChars = len(generator.charSet.getAll())
    while len(generator.positions) > 0:
        pos = generator.positions.pop(0)
        if pos is None:
            continue
        row, col = pos
        startX, startY, endX, endY = getSliceBounds(generator, row, col)
        generator.comboGrid.put(row, col, np.random.randint(1, numChars + 1))
        generator.mockupImg[startY:endY, startX:endX] = compositeAdj(
            generator, row, col
        )


def compare_amse(im1, im2, asymmetry):
    # Based on source of skimage compare_mse
    # https://github.com/scikit-image/scikit-image/blob/master/skimage/measure/simple_metrics.py#L27
    def _assert_compatible(im1, im2):
        if not im1.shape == im2.shape:
            raise ValueError("Input images must have the same dimensions.")
        return

    def _as_floats(im1, im2):
        float_type = np.result_type(im1.dtype, im2.dtype, np.float32)
        im1 = np.asarray(im1, dtype=float_type)
        im2 = np.asarray(im2, dtype=float_type)
        return im1, im2

    _assert_compatible(im1, im2)
    im1, im2 = _as_floats(im1, im2)
    diff = im1 - im2
    result = np.where(diff > 0, diff * (1 + asymmetry), diff)
    return np.mean(np.square(result), dtype=np.float64)


# Uses combos to store already composited "full" (all 4 layers)
# If combo not already generated, add it to comboSet.
# Returns mockupImg slice
def compositeAdj(
    generator,
    row,
    col,
    shrunken=False,
    targetSlice="",
    addFixed=True,
    subtractive=False,
):
    def getIndices(cDict):
        t = cDict[0], cDict[1], cDict[2], cDict[3]
        # print(cDict)
        return t

    def getChars(cDict):
        return (
            generator.charSet.getByID(cDict[0]),
            generator.charSet.getByID(cDict[1]),
            generator.charSet.getByID(cDict[2]),
            generator.charSet.getByID(cDict[3]),
        )

    qs = {}  # Quadrants

    # TL , TR, BL, BR
    for posID in [0, 1, 2, 3]:
        aRow = row
        aCol = col
        if posID in [1, 3]:
            aCol += 1
        if posID in [2, 3]:
            aRow += 1
        idx = getIndices(generator.comboGrid.get(aRow, aCol))
        combo = generator.comboSet.getCombo(*idx)
        if not combo:
            # Combo not found
            chars = getChars(generator.comboGrid.get(aRow, aCol))
            combo = generator.comboSet.genCombo(*chars)
        qs[posID] = combo

    # Stitch quadrants together
    startX, startY, endX, endY = getSliceBounds(generator, row, col, shrunken=False)
    img = generator.fixedMockupImg[startY:endY, startX:endX] / 255
    if not addFixed:
        img.fill(1)
    img[: img.shape[0] // 2, : img.shape[1] // 2] *= qs[0].img
    img[: img.shape[0] // 2, img.shape[1] // 2 :] *= qs[1].img
    img[img.shape[0] // 2 :, : img.shape[1] // 2] *= qs[2].img
    img[img.shape[0] // 2 :, img.shape[1] // 2 :] *= qs[3].img
    if subtractive:
        scale = generator.subtractiveScale
        typed = np.array(img * 255, dtype="uint8")
        typed = cv2.subtract(255, cv2.add(typed, scale))
        targetSlice = cv2.subtract(255, targetSlice)
        targetSlice = cv2.subtract(targetSlice, typed)
        return cv2.subtract(255, targetSlice)
    if shrunken:
        img = cv2.resize(
            img,
            dsize=(generator.shrunkenComboW * 2, generator.shrunkenComboH * 2),
            interpolation=cv2.INTER_AREA,
        )
    return np.array(img * 255, dtype="uint8")


def evaluateMockup(generator):
    psnr = compare_psnr(generator.mockupImg, generator.targetImg)
    ssim = compare_ssim(generator.mockupImg, generator.targetImg)
    return psnr, ssim
