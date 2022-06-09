import numpy as np
import matplotlib.pyplot as plt
import math
import random
from timeit import Timer
from annoy import AnnoyIndex
from matplotlib.image import imread, imsave
from PIL import Image
import cv2
from math import inf
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import normalized_root_mse as compare_nrmse
import sys
from scipy.ndimage import sobel
import json
import time

from combo import Combo, ComboSet
from combo_grid import ComboGrid
from char import Char, CharSet
from generator import Generator
from kword_utils import chop_charset, resizeTarget, genMockup

# Colab specific

def kword(
    basePath = '',
    sourceFn = 'sc-3tone.png',
    slicesX = 50,
    slicesY = 34,
    xPad = 0,
    yPad = 0,
    targetFn = 'maisie-williams.png',
    rowLength = 20,
    c = 1,
    shrinkX = 1,
    shrinkY = 1,
    mode = 'amse',
    gamma = 1,
    resume = False,
    numAdjust = 1,
    randomInit = False,
    randomOrder = False,
    autoCrop = False,
    crop = False,
    zoom = 0,
    shiftLeft = 0,
    shiftUp = 0,
    show = True,
    initMode = 'blend',
    initOnly = False,
    saveChars = False,
    initPriority = False,
    initComposite = False,
    genLayers = False,
    initBrighten = 0,
    asymmetry = 0.1,
    search = 'greedy',
    maxVisits = 5,
    printEvery = 10,
    initTemp = 10,
    tempStep = 0.001,
    scaleTemp = 1,
    initK = 5,
    subtractiveScale = 128,
    selectOrder = 'linear',
    blendFunc = '2*amse + ssim',
    whiteThreshold = 0.95,
    excludeChars = [],
    tempReheatFactor = 0.5,
    hiResMockup = True,
    numLayers = 2,
    maxChars = 5790
    ):

    config = locals()
    # args = sys.argv
    startTime = time.time()
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
        fn=basePath+sourceFn, numX=slicesX, numY=slicesY, startX=0, startY=0,
        xPad=xPad, yPad=yPad, shrinkX=shrinkX, shrinkY=shrinkY, blankSpace=True,
        whiteThreshold=whiteThreshold, basePath=basePath, excludeChars=excludeChars)
    cropSettings = {
        'xPad': xPad,
        'yPad': yPad,
        'xCropPos': xCropPos,
        'yCropPos': yCropPos,
        'shrinkX': shrinkX,
        'shrinkY': shrinkY
    }
    charSet = CharSet(padded, cropSettings, excludeChars)
    # mockupFn = f"mockup/mp_{targetFn.split('.')[-2]}_{rowLength}_{mode}"


    targetImg = cv2.imread(basePath+targetFn, cv2.IMREAD_GRAYSCALE)
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


    # Autocrop routine, terminates the rest of the program
    if autoCrop:
        annSaved = False
        charHeight, charWidth = charSet.get(0).cropped.shape

        scores = []
        # Try 2 zooms: none, and add 1 quadrant to bottom and right sides
        for zoom in range(4):
            # Shift 0-1.5 quadrants
            for shiftLeft in range(4):
                shiftLeft = int(((charWidth / 4) * shiftLeft) / 2)
                for shiftUp in range(4):
                    shiftUp = int(((charHeight / 4) * shiftUp) / 2)
                    resizedTarget, targetPadding, rowLength = resizeTarget(targetImg, rowLength, charSet.get(0).cropped.shape, (xChange, yChange), numLayers, maxChars)
                    origShape = resizedTarget.shape
                    height = (resizedTarget.shape[0] + zoom * charHeight//4)
                    width = (resizedTarget.shape[1] + zoom * charWidth//4)
                    # zoom target
                    resizedTarget = cv2.resize(resizedTarget, dsize=(width, height), interpolation=cv2.INTER_AREA)
                    # shift target left and up
                    resizedTarget = resizedTarget[shiftUp:,shiftLeft:]
                    newTarget = np.full(origShape, 255, dtype='uint8')
                    # crop or pad
                    if resizedTarget.shape[0] >= origShape[0]:
                        if resizedTarget.shape[1] >= origShape[1]:
                            # crop right and bottom
                            newTarget[:, :] = resizedTarget[:origShape[0], :origShape[1]]
                            minShape = newTarget.shape
                        # pad right, crop bottom
                        else:
                            newTarget[:, :resizedTarget.shape[1]] = resizedTarget[:origShape[0], :]
                            minShape = [origShape[0], resizedTarget.shape[1]]
                    else:
                        if resizedTarget.shape[1] >= origShape[1]:
                            # crop right, pad bottom
                            newTarget[:resizedTarget.shape[0], :] = resizedTarget[:, :origShape[1]]
                            minShape = [resizedTarget.shape[0], origShape[1]]
                        else:
                            # pad right and bottom
                            newTarget[:resizedTarget.shape[0], :resizedTarget.shape[1]] = resizedTarget[:origShape[0], :]
                            minShape = resizedTarget.shape
                    #################################################
                    # Generate mockup (the part that really matters!)
                    generator = Generator(newTarget, newTarget, charSet, targetShape=targetImg.shape,
                                             targetPadding=targetPadding, shrunkenTargetPadding=targetPadding,
                                             subtractiveScale=subtractiveScale, basePath=basePath)
                    if annSaved:
                        generator.loadAnn()
                    else:
                        # Build angular and euclidean ANN models
                        generator.buildAnn()
                        annSaved = True
                    # THIS IS THE LINE THAT MATTERS
                    generator.generateLayers(compareMode=mode, numAdjustPasses=numAdjust, gamma=gamma, 
                        show=show, init=initMode, initOnly=True,
                        initPriority=initPriority, initComposite=initComposite, search=search)
                    ###################
                    # Save init image
                    mockupFn = f'{basePath}results/init_{rowLength}_{zoom}_{shiftLeft}_{shiftUp}'
                    print("writing init file: ",mockupFn)
                    mockupImg = generator.mockupImg
                    # Crop added whitespace from shifting
                    mockupImg = mockupImg[:minShape[0], :minShape[1]]
                    newTarget = newTarget[:minShape[0], :minShape[1]]
                    psnr = compare_psnr(mockupImg, newTarget)
                    ssim = compare_ssim(mockupImg, newTarget)
                    print("PSNR:", psnr)
                    print("SSIM:", ssim)
                    cv2.imwrite(mockupFn+'.png', mockupImg)
                    scores.append((ssim+psnr, ssim, psnr, mockupFn))

        scores = sorted(scores, reverse=True)
        with open(f"{basePath}results/autocropScores.json","w") as f:
            f.write(json.dumps(scores))
        for score in scores:
            print(score)
            
        return

    # else:
    #     shrunkenTarget, shrunkenTargetPadding = resizeTarget(targetImg, rowLength, charSet.get(0).shrunken.shape, (xChange, yChange))
    #     print('shrunken char shape', charSet.get(0).shrunken.shape)
    #     # resizedCharShape = charSet.get(0).shrunken.shape[0] * shrinkY, charSet.get(0).shrunken.shape[1] * shrinkX
    #     resizedTarget, targetPadding = resizeTarget(targetImg, rowLength, charSet.get(0).cropped.shape, (xChange, yChange))
    #     print('shrunkenTarget.shape', shrunkenTarget.shape)
    #     print('resizedTarget.shape', resizedTarget.shape)
    charHeight, charWidth = charSet.get(0).cropped.shape
    resizedTarget, targetPadding, rowLength = resizeTarget(targetImg, rowLength, charSet.get(0).cropped.shape, (xChange, yChange), numLayers, maxChars)
    print("new row length:", rowLength)
    origShape = resizedTarget.shape
    height = (resizedTarget.shape[0] + zoom * charHeight//4)
    width = (resizedTarget.shape[1] + zoom * charWidth//4)
    # zoom target
    resizedTarget = cv2.resize(resizedTarget, dsize=(width, height), interpolation=cv2.INTER_AREA)
    # shift target left and up
    resizedTarget = resizedTarget[shiftUp:,shiftLeft:]
    newTarget = np.full(origShape, 255, dtype='uint8')
    # crop or pad
    if resizedTarget.shape[0] >= origShape[0]:
        if resizedTarget.shape[1] >= origShape[1]:
            # crop right and bottom
            newTarget[:, :] = resizedTarget[:origShape[0], :origShape[1]]
            minShape = newTarget.shape
        # pad right, crop bottom
        else:
            newTarget[:, :resizedTarget.shape[1]] = resizedTarget[:origShape[0], :]
            minShape = [origShape[0], resizedTarget.shape[1]]
    else:
        if resizedTarget.shape[1] >= origShape[1]:
            # crop right, pad bottom
            newTarget[:resizedTarget.shape[0], :] = resizedTarget[:, :origShape[1]]
            minShape = [resizedTarget.shape[0], origShape[1]]
        else:
            # pad right and bottom
            newTarget[:resizedTarget.shape[0], :resizedTarget.shape[1]] = resizedTarget[:origShape[0], :]
            minShape = resizedTarget.shape
    #################################################
    # Generate mockup (the part that really matters!)
    generator = Generator(newTarget, newTarget, charSet, targetShape=targetImg.shape,
                                targetPadding=targetPadding, shrunkenTargetPadding=targetPadding,
                                asym=asymmetry, initTemp=initTemp, initK=initK,
                                subtractiveScale=subtractiveScale, selectOrder=selectOrder,
                                basePath=basePath, tempStep=tempStep, scaleTemp=scaleTemp,
                                blendFunc=blendFunc, tempReheatFactor=tempReheatFactor)
    # if annSaved:
    #     generator.loadAnn()
    # else:
    #     # Build angular and euclidean ANN models
    #     generator.buildAnn()
    #     annSaved = True
    # #################################################
    # # Generate mockup (the part that really matters!)
    # generator = Generator(resizedTarget, shrunkenTarget, charSet, targetShape=targetImg.shape,
    #                                     targetPadding=targetPadding, shrunkenTargetPadding=shrunkenTargetPadding)

    if resume != False:
        if type(resume) == type('hi'):
            generator.load_state(basePath + resume)
        else:
            generator.load_state()

    # Build angular and euclidean ANN models
    generator.buildAnn()
    # if resume != False:
    #     initMode = None
    # THIS IS THE LINE THAT MATTERS
    generator.generateLayers(compareMode=mode, numAdjustPasses=numAdjust, gamma=gamma, 
                        show=show, init=initMode, initOnly=initOnly,
                        initPriority=initPriority, initComposite=initComposite, search=search,
                        maxVisits=maxVisits, printEvery=printEvery)

    # print(generator.comboGrid)
    # plt.style.use('default')
    # plt.axis('off')
    # plt.imshow(generator.mockupImg, cmap='gray', vmin=0, vmax=255)
    # plt.show()

    #!!! Download zip of each run including settings, typeables, mockup, comboGrid, graphs, data for graphs

    ###################

    # Save mockup image
    mockupImg = generator.mockupImg
    if targetPadding > 0: # Crop and resize mockup to match target image
        mockupImg = mockupImg[:-targetPadding, :]
    if hiResMockup:
        # Correct aspect
        newHeight = int(mockupImg.shape[1] * targetImg.shape[0]/targetImg.shape[1])
        resized = cv2.resize(mockupImg, dsize=(mockupImg.shape[1], newHeight), interpolation=cv2.INTER_AREA)
    else:
        # Match to target image
        resized = cv2.resize(mockupImg, dsize=(targetImg.shape[1],targetImg.shape[0]), interpolation=cv2.INTER_AREA)
    cv2.imwrite(basePath+'results/mockup_optimized.png', resized)

    # Save config
    with open(f"{basePath}results/config.json","w") as f:
        f.write(json.dumps(config))

    # Save combogrid
    np.savetxt(f"{basePath}results/grid_optimized.txt", generator.comboGrid.getPrintable(), fmt='%i', delimiter=" ")

    # Save layer images
    if genLayers:
        layerNames = ['BR', 'BL', 'TR', 'TL']
        for i, layer in enumerate(generator.comboGrid.getLayers()):
            layerImg = genMockup(layer, generator, targetImg.shape, targetPadding, crop=False, addFixed=False)
            cv2.imwrite(f"{basePath}results/typeable_{layerNames[i]}.png", layerImg)

    # Save score history
    # TODO make this nices (how many positions optimized?)
    with open(f"{basePath}results/history_scores.txt", 'w') as f:
        f.write('\n'.join([f'{s[0]} {s[1]} {s[2]}' for s in generator.psnrHistory]))

    # Save choice history
    with open(f"{basePath}results/history_choices.txt", 'w') as f:
        f.write('Row Col ChosenID\n')
        f.write('\n'.join([str(c[0])+' '+str(c[1])+' '+str(c[2]) for c in generator.choiceHistory]))

    ############################
    # Calculate scores on result, print and save
    stats = f"{generator.frame} positions optimized\n{generator.stats['comparisonsMade']} comparisons made\n{time.time()-startTime:.2f} seconds"

    print(stats)

    with open(f"{basePath}results/history_stats.txt", 'w') as f:
        f.write(stats)
    
    # Overlay the original image for comparison
    # cv2.imwrite(mockupFn+'c.png', cv2.addWeighted(resized,0.5,targetImg,0.5,0))