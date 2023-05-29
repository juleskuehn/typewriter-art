# Generator

import numpy as np
import operator
import timeit
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib import animation, rc, ticker
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
import pickle
import sys
from math import ceil
import concurrent.futures

from combo import ComboSet, Combo
from combo_grid import ComboGrid
from generator_utils import *
from queues import LinearQueue

from IPython.display import display, clear_output
import warnings

warnings.filterwarnings("ignore")


def save_object(obj, filename):
    with open(filename, "wb") as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, "rb") as input:
        return pickle.load(input)


def updateProgress(generator):
    # totalVisitable = (maxVisits + 1) * (generator.rows-2) * (generator.cols-2)
    # numVisited = np.sum(generator.comboGrid.flips[1:-1, 1:-1])
    scores = evaluateMockup(generator)
    # lastScores = generator.psnrHistory[-1][1:]
    # psnrDiff = scores[0] - lastScores[0]
    # ssimDiff = scores[1] - lastScores[1]
    # scoresStr = f"PSNR: {scores[0]:2.2f} ({'+' if psnrDiff >= 0 else ''}{psnrDiff:2.2f}) SSIM: {scores[1]:0.4f} ({'+' if ssimDiff >= 0 else ''}{ssimDiff:0.4f})"
    # progressBar(numVisited, totalVisitable, scoresStr)
    generator.psnrHistory.append([generator.frame, scores[0], scores[1]])
    generator.tempHistory.append(generator.temperature)
    # print("Temperature =", generator.temperature)

    # Save the results so far
    np.savetxt(
        f"{generator.basePath}results/grid_optimized.txt",
        generator.comboGrid.getPrintable(),
        fmt="%i",
        delimiter=" ",
    )
    # Save choice history
    with open(f"{generator.basePath}results/history_choices.txt", "w") as f:
        f.write("Row Col ChosenID\n")
        f.write(
            "\n".join(
                [
                    str(c[0]) + " " + str(c[1]) + " " + str(c[2])
                    for c in generator.choiceHistory
                ]
            )
        )


def updateGraphs(generator, save=False):
    # Mockup image
    plt.subplot(121)
    plt.style.use("default")
    plt.axis("off")
    plt.imshow(generator.mockupImg, cmap="gray", vmin=0, vmax=255)

    # Scores / number optimized
    # TODO: 2 y-axis labels (SSIM, PSNR)
    plt.subplot(122)
    plt.style.use("default")
    normScores = [
        (mse, ssim * 45, generator.tempHistory[i])
        for i, [_, mse, ssim] in enumerate(generator.psnrHistory)
    ]
    [a, b, c] = plt.plot(normScores)

    # Set xticks appropriately
    ax = plt.gca()
    ticks = ticker.FuncFormatter(
        lambda x, pos: "{0:g}".format(x * generator.printEvery)
    )
    ax.xaxis.set_major_formatter(ticks)

    ssimString = (f"{generator.psnrHistory[-1][2]:.4f}")[1:]
    plt.title(
        f"{generator.psnrHistory[-1][1]:.2f} {ssimString} {100.*generator.randomChoices/generator.printEvery:.1f}"
    )
    generator.randomChoices = 0

    plt.legend([a, b, c], ["PSNR", "SSIM*45", "Temp"], loc=0)
    if save:
        plt.savefig(
            generator.basePath + "results/summary.png",
            dpi=None,
            facecolor="w",
            edgecolor="w",
            orientation="portrait",
            # papertype=None,
            format=None,
            transparent=False,
            bbox_inches=None,
            pad_inches=0,
            # frameon=None,
            metadata=None,
        )
    else:
        display(generator.fig)
        clear_output(wait=True)


class Generator:
    # Assumes targetImg has already been resized and padded to match combo dimensions
    def __init__(
        self,
        targetImg,
        shrunkenTargetImg,
        charSet,
        targetShape=None,
        targetPadding=None,
        shrunkenTargetPadding=None,
        asym=0.1,
        initTemp=0.005,
        subtractiveScale=64,
        selectOrder="linear",
        basePath="",
        tempStep=0.001,
        scaleTemp=1.0,
        blendFunc="2*amse + ssim",
        tempReheatFactor=0.5,
    ):
        self.asym = asym
        self.targetImg = targetImg
        # print('t', targetImg.shape)
        self.shrunkenTargetImg = shrunkenTargetImg
        # print('s', shrunkenTargetImg.shape)
        self.charSet = charSet
        self.comboSet = ComboSet()
        self.comboH, self.comboW = (
            charSet.get(0).cropped.shape[0] // 2,
            charSet.get(0).cropped.shape[1] // 2,
        )
        self.shrunkenComboH, self.shrunkenComboW = (
            charSet.get(0).shrunken.shape[0] // 2,
            charSet.get(0).shrunken.shape[1] // 2,
        )
        self.mockupRows = targetImg.shape[0] // self.comboH
        self.mockupCols = targetImg.shape[1] // self.comboW
        # print('mockupRows', self.mockupRows, 'mockupCols', self.mockupCols)
        self.rows = shrunkenTargetImg.shape[0] // self.shrunkenComboH
        self.cols = shrunkenTargetImg.shape[1] // self.shrunkenComboW
        # print('rows      ', self.rows,       'cols      ', self.cols)
        self.targetShape = targetShape or targetImg.shape
        self.mockupImg = np.full(targetImg.shape, 255, dtype="uint8")
        self.fixedMockupImg = np.full(targetImg.shape, 255, dtype="uint8")
        self.shrunkenMockupImg = np.full(shrunkenTargetImg.shape, 255, dtype="uint8")
        self.targetPadding = targetPadding or 0
        self.shrunkenTargetPadding = shrunkenTargetPadding or 0
        self.comboGrid = ComboGrid(self.rows, self.cols)
        self.compareMode = "mse"
        self.numLayers = 0  # How many times has the image been typed
        self.overtype = 1  # How many times have all 4 layers been typed
        self.passNumber = 0
        self.stats = {"positionsVisited": 0, "comparisonsMade": 0}
        self.initTemp = initTemp
        self.temperature = initTemp
        self.tempStep = tempStep
        self.tempReheatFactor = tempReheatFactor
        self.minTemp = 0.00001
        self.tempHistory = []
        self.psnrHistory = []
        self.randomChoices = 0
        self.positions = []
        self.subtractiveScale = subtractiveScale
        self.selectOrder = selectOrder
        self.choiceHistory = []
        self.basePath = basePath
        self.frame = 0  # Number of positions visited
        self.scaleTemp = scaleTemp
        self.blendFunc = eval("lambda amse, ssim:" + blendFunc)
        self.scores = np.zeros((self.rows, self.cols))

    def load_state(self, fn=None):
        if fn == None:
            fn = f"{self.basePath}results/resume.pkl"
        print("loaded state from file", fn)
        state = load_object(fn)
        self.fixedMockupImg = state["mockupImg"]
        self.mockupImg = state["mockupImg"].copy()

    def generateLayers(
        self,
        compareMode="mse",
        numAdjustPasses=0,
        show=True,
        mockupFn="mp_untitled",
        init="blank",
        initOnly=False,
        initPriority=False,
        initComposite=False,
        initBrighten=0,
        search="greedy",
        maxVisits=5,
        printEvery=10,
    ):
        self.printEvery = printEvery

        def setupFig():
            fig, ax = plt.subplots(figsize=(8, 5))
            # plt.rcParams['figure.figsize'] = [12, 3]
            return fig, ax

        self.compareMode = compareMode

        self.fig, self.ax = setupFig()

        if init == "random":
            for row in range(self.rows - 1):
                for col in range(self.cols - 1):
                    position = (row, col)
                    self.positions.append(position)
            initRandomPositions(self)

        # Otherwise init is blank

        # Save initial combogrid
        np.savetxt(
            f"{self.basePath}results/grid_initial.txt",
            self.comboGrid.getPrintable(),
            fmt="%i",
            delimiter=" ",
        )

        # Save initial mockup
        mockupImg = self.mockupImg.copy()
        if self.targetPadding > 0:  # Crop and resize mockup to match target image
            mockupImg = mockupImg[: -self.targetPadding, :]

        resized = cv2.resize(
            mockupImg,
            dsize=(self.targetShape[1], self.targetShape[0]),
            interpolation=cv2.INTER_AREA,
        )
        cv2.imwrite(self.basePath + "results/mockup_initial.png", resized)

        def progressBar(value, endvalue, scores, bar_length=20):
            percent = float(value) / endvalue
            arrow = "-" * int(round(percent * bar_length) - 1) + ">"
            spaces = " " * (bar_length - len(arrow))
            sys.stdout.write(
                "\rPercent: [{0}] {1}% {2}".format(
                    arrow + spaces, int(round(percent * 100)), scores
                )
            )
            sys.stdout.flush()

        self.frame = 0
        # updateProgress()

        if not initOnly:
            # self.psnrHistory.append(evaluateMockup(self))
            if self.selectOrder == "random":
                self.queue = LinearQueue(self, maxVisits=maxVisits, randomOrder=True)
            else:
                self.queue = LinearQueue(self, maxVisits=maxVisits, randomOrder=False)

            while True:
                shouldBreak = optimizationLoop(self)
                if shouldBreak:
                    break

        # This only runs when the function completes
        updateProgress(self)
        updateGraphs(self)
        updateGraphs(self, save=True)

        # print("\n", frame, 'positions visited,', self.stats['comparisonsMade'], 'comparisons made')

        save_object(
            {"mockupImg": self.mockupImg, "comboGrid": self.comboGrid},
            f"{self.basePath}results/resume.pkl",
        )

        print(self.temperature)


def optimizationLoop(generator):
    # Stopping condition: no change over past (lots of) scores
    # Stopping condition: keyboard interrupt in try/catch
    try:
        # Stopping condition: maxFlips reached
        try:
            pos, bestId = generator.queue.remove()
            # LinearQueue doesn't return a bestId
            # We only need bestId for greedy search
            row, col = pos
        except:
            return True

        if generator.frame % generator.printEvery == 0:
            updateProgress(generator)
            updateGraphs(generator)
        generator.frame += 1
        row, col = pos
        generator.comboGrid.flips[row, col] += 1

        choice = getSimAnneal(generator, row, col)

        if choice != None and choice != generator.comboGrid.get(row, col)[3]:
            generator.comboGrid.put(row, col, choice, chosen=True)
            startX, startY, endX, endY = getSliceBounds(
                generator, row, col, shrunken=False
            )
            if row < generator.mockupRows - 1 and col < generator.mockupCols - 1:
                generator.mockupImg[startY:endY, startX:endX] = compositeAdj(
                    generator, row, col, shrunken=False
                )
            if generator.selectOrder == "priority":
                generator.queue.update(row, col)
            # Save choice
            generator.choiceHistory.append([row, col, choice])

        generator.queue.add((row, col))

        generator.temperature -= generator.tempStep
        if generator.temperature <= generator.minTemp:
            generator.initTemp *= generator.tempReheatFactor
            generator.tempStep *= generator.tempReheatFactor
            generator.temperature = generator.initTemp

    except KeyboardInterrupt:
        return True
    return False
