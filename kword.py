import numpy as np
import cv2
import json
import time
import pickle

from char import CharSet
from generator import Generator
from kword_utils import chop_charset, resizeTarget, genMockup

# Colab specific


def kword(
    basePath="",
    sourceFn="sc-3tone.png",
    slicesX=50,
    slicesY=34,
    xPad=0,
    yPad=0,
    targetFn="maisie-williams.png",
    rowLength=20,
    c=1,
    shrinkX=1,
    shrinkY=1,
    mode="amse",
    resume=False,
    numAdjust=1,
    randomInit=False,
    randomOrder=False,
    autoCrop=False,
    crop=False,
    zoom=0,
    shiftLeft=0,
    shiftUp=0,
    show=True,
    initMode="blend",
    initOnly=False,
    saveChars=False,
    initPriority=False,
    initComposite=False,
    genLayers=False,
    initBrighten=0,
    asymmetry=0.1,
    search="greedy",
    maxVisits=5,
    printEvery=10,
    initTemp=10,
    tempStep=0.001,
    scaleTemp=1.0,
    subtractiveScale=128,
    selectOrder="linear",
    blendFunc="2*amse + ssim",
    whiteThreshold=0.95,
    excludeChars=[],
    tempReheatFactor=0.5,
    hiResMockup=True,
    numLayers=2,
    maxChars=5790,
):
    config = locals()
    # args = sys.argv
    # Hardcoding the charset params for convenience

    # sourceFn = 'marker-shapes.png'
    # slicesX = 12
    # slicesY = 2
    # xPad = 0
    # yPad = 0

    # sourceFn = 'sc-3toneNew.png'
    # slicesX = 45
    # slicesY = 21
    # xPad = 0
    # yPad = 0

    # sourceFn = 'sc-3toneNew2.png'
    # slicesX = 45
    # slicesY = 25
    # xPad = 0
    # yPad = 0

    # sourceFn = 'sc-3tone.png'
    # slicesX = 50
    # slicesY = 34
    # xPad = 0
    # yPad = 0

    # sourceFn = 'sc-1tone.png'
    # slicesX = 26
    # slicesY = 15
    # xPad = 0
    # yPad = 0

    # sourceFn = 'hermes-darker.png'
    # slicesX = 79
    # slicesY = 7
    # xPad = 4
    # yPad = 4

    #################
    # Prepare charset
    cropped, padded, (xCropPos, yCropPos), (xChange, yChange) = chop_charset(
        fn=basePath + sourceFn,
        numX=slicesX,
        numY=slicesY,
        startX=0,
        startY=0,
        xPad=xPad,
        yPad=yPad,
        shrinkX=shrinkX,
        shrinkY=shrinkY,
        blankSpace=True,
        whiteThreshold=whiteThreshold,
        basePath=basePath,
        excludeChars=excludeChars,
    )
    cropSettings = {
        "xPad": xPad,
        "yPad": yPad,
        "xCropPos": xCropPos,
        "yCropPos": yCropPos,
        "shrinkX": shrinkX,
        "shrinkY": shrinkY,
    }
    charSet = CharSet(padded, cropSettings, excludeChars)
    # mockupFn = f"mockup/mp_{targetFn.split('.')[-2]}_{rowLength}_{mode}"

    targetImg = cv2.imread(basePath + targetFn, cv2.IMREAD_GRAYSCALE)
    #     print("target photo has shape", targetImg.shape)

    # Save characters
    # import os
    # d = f'{basePath}results/chars/'
    # filesToRemove = [os.path.join(d,f) for f in os.listdir(d)]
    # for f in filesToRemove:
    #     os.remove(f)
    # if saveChars:
    #     for i, char in enumerate(charSet.getAll()):
    #         cv2.imwrite(f'{basePath}results/chars/{char.id}.png', char.cropped)

    # else:
    #     shrunkenTarget, shrunkenTargetPadding = resizeTarget(targetImg, rowLength, charSet.get(0).shrunken.shape, (xChange, yChange))
    #     print('shrunken char shape', charSet.get(0).shrunken.shape)
    #     # resizedCharShape = charSet.get(0).shrunken.shape[0] * shrinkY, charSet.get(0).shrunken.shape[1] * shrinkX
    #     resizedTarget, targetPadding = resizeTarget(targetImg, rowLength, charSet.get(0).cropped.shape, (xChange, yChange))
    #     print('shrunkenTarget.shape', shrunkenTarget.shape)
    #     print('resizedTarget.shape', resizedTarget.shape)
    charHeight, charWidth = charSet.get(0).cropped.shape
    resizedTarget, targetPadding, rowLength = resizeTarget(
        targetImg,
        rowLength,
        charSet.get(0).cropped.shape,
        (xChange, yChange),
        numLayers,
        maxChars,
    )
    print("new row length:", rowLength)
    origShape = resizedTarget.shape
    height = resizedTarget.shape[0] + zoom * charHeight // 4
    width = resizedTarget.shape[1] + zoom * charWidth // 4
    # zoom target
    resizedTarget = cv2.resize(
        resizedTarget, dsize=(width, height), interpolation=cv2.INTER_AREA
    )
    # shift target left and up
    resizedTarget = resizedTarget[shiftUp:, shiftLeft:]
    newTarget = np.full(origShape, 255, dtype="uint8")
    # crop or pad
    if resizedTarget.shape[0] >= origShape[0]:
        if resizedTarget.shape[1] >= origShape[1]:
            # crop right and bottom
            newTarget[:, :] = resizedTarget[: origShape[0], : origShape[1]]
            minShape = newTarget.shape
        # pad right, crop bottom
        else:
            newTarget[:, : resizedTarget.shape[1]] = resizedTarget[: origShape[0], :]
            minShape = [origShape[0], resizedTarget.shape[1]]
    else:
        if resizedTarget.shape[1] >= origShape[1]:
            # crop right, pad bottom
            newTarget[: resizedTarget.shape[0], :] = resizedTarget[:, : origShape[1]]
            minShape = [resizedTarget.shape[0], origShape[1]]
        else:
            # pad right and bottom
            newTarget[
                : resizedTarget.shape[0], : resizedTarget.shape[1]
            ] = resizedTarget[: origShape[0], :]
            minShape = resizedTarget.shape

    # # 2023 rewrite: pad target so it has a border of 1/2 character width/height on every side
    newTarget = np.pad(
        newTarget,
        ((charHeight // 2, charHeight // 2), (charWidth // 2, charWidth // 2)),
        "constant",
        constant_values=255,
    )
    # If the bottom charHeight rows are fully white, remove them
    if np.all(newTarget[-charHeight:, :] == 255):
        newTarget = newTarget[:-charHeight, :]

    ################
    # 2023 rewrite #
    ################
    num_rows = newTarget.shape[0] // charHeight
    num_cols = newTarget.shape[1] // charWidth
    # layer_offsets = [
    #     (0, 0),
    #     (0, charHeight // 2),
    #     (charWidth // 2, 0),
    #     (charWidth // 2, charHeight // 2),
    # ]
    layer_offsets = [(0, 0), (0, 0), (0, 0), (0, 0)]  # Keep it simple to test
    num_loops = 100

    # Internal representations of images should be float32 for easy math
    target = newTarget.astype("float32") / 255
    chars = np.array([c.cropped for c in charSet.chars], dtype="float32") / 255
    assert target.shape[0] % chars.shape[1] == 0
    assert target.shape[1] % chars.shape[2] == 0
    mockup = np.full(target.shape, 1, dtype="float32")

    # Layers is a 3D array of mockups, one per layer
    layers = np.array([mockup.copy() for _ in layer_offsets], dtype="float32")

    # Choices is a 3D array of indices for chars selected per layer
    choices = np.zeros((layers.shape[0], num_rows * num_cols), dtype="uint16")

    # Padded chars are only used for user-facing mockups - they can stay as uint8
    # padded_chars = np.array([c.padded for c in charSet.chars], dtype="uint8")
    # if resume:
    #     with open(f"{basePath}results/resume.pkl", "rb") as input:
    #         state = pickle.load(input)
    #     print("loaded resume state (background image)")
    #     layers.push(state["mockupImg"])

    startTime = time.perf_counter_ns()
    # Layer optimization passes
    for loop_num in range(num_loops):
        for layer_num, layer_offset in enumerate(layer_offsets):
            # Composite all other layers
            bg = np.prod(np.delete(layers, layer_num, axis=0), axis=0)

            choices[layer_num], mockup = layer_optimization_pass(
                bg,
                mockup,
                target,
                chars,
                choices[layer_num],
                layer_offset,
            )

            # Update the layer image based on the returned choices
            for i, choice in enumerate(choices[layer_num]):
                row = i // num_cols
                col = i % num_cols
                replace_this_slice = layers[layer_num][
                    row * chars.shape[1]
                    + layer_offset[0] : (row + 1) * chars.shape[1]
                    + layer_offset[0],
                    col * chars.shape[2]
                    + layer_offset[1] : (col + 1) * chars.shape[2]
                    + layer_offset[1],
                ]
                if replace_this_slice.shape != chars[choice].shape:
                    continue
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
                f"{basePath}results/mockup_{loop_num}_{layer_num}.png", mockupImg
            )

    endTime = time.perf_counter_ns()
    print(f"Layer optimization took {(endTime - startTime) / 1e9} seconds")
    return


def layer_optimization_pass(
    bg,
    mockup,
    target,
    chars,
    choices,
    layer_offset,
    asymmetry=0.1,
    mode="greedy",
    temperature=0.1,
):
    num_cols = target.shape[1] // chars.shape[2]

    # Pad the images with white by the layer_offset
    target, mockup, bg = (
        np.pad(img, ((layer_offset[0], 0), (layer_offset[1], 1)), "constant")
        for img in (target, mockup, bg)
    )

    for i, prev_choice in enumerate(choices):
        row = i // num_cols
        col = i % num_cols
        # Get the slices of target, mockup and background for this position
        target_slice, mockup_slice, bg_slice = (
            img[
                row * chars.shape[1]
                + layer_offset[0] : (row + 1) * chars.shape[1]
                + layer_offset[0],
                col * chars.shape[2]
                + layer_offset[1] : (col + 1) * chars.shape[2]
                + layer_offset[1],
            ]
            for img in (target, mockup, bg)
        )
        # Get the character that was previously chosen for this position
        cur_composite = mockup_slice
        err = target_slice - cur_composite
        asym_err = np.where(err > 0, err * (1 + asymmetry), err)
        cur_amse = np.mean(np.square(asym_err))
        # Try other characters in this position
        for new_choice in range(chars.shape[0]):
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
            elif mode == "sim_anneal":
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

    return choices, mockup


def old_stuff():
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
