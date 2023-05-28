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

from combo import ComboSet, Combo
from combo_grid import ComboGrid
from generator_utils import *
from queues import PositionQueue, LinearQueue

from IPython.display import display, clear_output
import warnings

warnings.filterwarnings("ignore")


def save_object(obj, filename):
    with open(filename, "wb") as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, "rb") as input:
        return pickle.load(input)


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

        fig, ax = setupFig()

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

        def updateProgress():
            # totalVisitable = (maxVisits + 1) * (self.rows-2) * (self.cols-2)
            # numVisited = np.sum(self.comboGrid.flips[1:-1, 1:-1])
            scores = evaluateMockup(self)
            # lastScores = self.psnrHistory[-1][1:]
            # psnrDiff = scores[0] - lastScores[0]
            # ssimDiff = scores[1] - lastScores[1]
            # scoresStr = f"PSNR: {scores[0]:2.2f} ({'+' if psnrDiff >= 0 else ''}{psnrDiff:2.2f}) SSIM: {scores[1]:0.4f} ({'+' if ssimDiff >= 0 else ''}{ssimDiff:0.4f})"
            # progressBar(numVisited, totalVisitable, scoresStr)
            self.psnrHistory.append([self.frame, scores[0], scores[1]])
            self.tempHistory.append(self.temperature)
            # print("Temperature =", self.temperature)

            # Save the results so far
            np.savetxt(
                f"{self.basePath}results/grid_optimized.txt",
                self.comboGrid.getPrintable(),
                fmt="%i",
                delimiter=" ",
            )
            # Save choice history
            with open(f"{self.basePath}results/history_choices.txt", "w") as f:
                f.write("Row Col ChosenID\n")
                f.write(
                    "\n".join(
                        [
                            str(c[0]) + " " + str(c[1]) + " " + str(c[2])
                            for c in self.choiceHistory
                        ]
                    )
                )

        def updateGraphs(fig, ax, save=False):
            # Mockup image
            plt.subplot(121)
            plt.style.use("default")
            plt.axis("off")
            plt.imshow(self.mockupImg, cmap="gray", vmin=0, vmax=255)

            # Scores / number optimized
            # TODO: 2 y-axis labels (SSIM, PSNR)
            plt.subplot(122)
            plt.style.use("default")
            normScores = [
                (mse, ssim * 45, self.tempHistory[i])
                for i, [_, mse, ssim] in enumerate(self.psnrHistory)
            ]
            [a, b, c] = plt.plot(normScores)

            # Set xticks appropriately
            ax = plt.gca()
            ticks = ticker.FuncFormatter(
                lambda x, pos: "{0:g}".format(x * self.printEvery)
            )
            ax.xaxis.set_major_formatter(ticks)

            ssimString = (f"{self.psnrHistory[-1][2]:.4f}")[1:]
            plt.title(
                f"{self.psnrHistory[-1][1]:.2f} {ssimString} {100.*self.randomChoices/printEvery:.1f}"
            )
            self.randomChoices = 0

            plt.legend([a, b, c], ["PSNR", "SSIM*45", "Temp"], loc=0)
            if save:
                plt.savefig(
                    self.basePath + "results/summary.png",
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
                display(fig)
                clear_output(wait=True)

        self.frame = 0
        # updateProgress()

        if not initOnly:
            # self.psnrHistory.append(evaluateMockup(self))
            if self.selectOrder == "priority":
                self.queue = PositionQueue(self, maxVisits=maxVisits)
            elif self.selectOrder == "random":
                self.queue = LinearQueue(self, maxVisits=maxVisits, randomOrder=True)
            else:
                self.queue = LinearQueue(self, maxVisits=maxVisits, randomOrder=False)

            while True:
                # Stopping condition: no change over past (lots of) scores
                numPositions = self.rows * self.cols
                numStats = ceil(numPositions / printEvery)
                if len(self.psnrHistory) >= numStats:
                    last = self.psnrHistory[-1]
                    if all(
                        s[1] == last[1] and s[2] == last[2]
                        for s in self.psnrHistory[-numStats:-1]
                    ):
                        # Reheat simAnneal
                        if search == "simAnneal":
                            self.temperature = self.initTemp
                        else:
                            break
                # Stopping condition: keyboard interrupt in try/catch
                try:
                    # Stopping condition: maxFlips reached
                    try:
                        pos, bestId = self.queue.remove()
                        # LinearQueue doesn't return a bestId
                        # We only need bestId for greedy search
                        row, col = pos
                    except:
                        break

                    if self.frame % printEvery == 0:
                        updateProgress()
                        updateGraphs(fig, ax)
                    self.frame += 1
                    row, col = pos
                    self.comboGrid.flips[row, col] += 1

                    choice = getSimAnneal(self, row, col)

                    if choice != None and choice != self.comboGrid.get(row, col)[3]:
                        self.comboGrid.put(row, col, choice, chosen=True)
                        startX, startY, endX, endY = getSliceBounds(
                            self, row, col, shrunken=False
                        )
                        if row < self.mockupRows - 1 and col < self.mockupCols - 1:
                            self.mockupImg[startY:endY, startX:endX] = compositeAdj(
                                self, row, col, shrunken=False
                            )
                        if self.selectOrder == "priority":
                            self.queue.update(row, col)
                        # Save choice
                        self.choiceHistory.append([row, col, choice])

                    self.queue.add((row, col))

                    self.temperature -= self.tempStep
                    if self.temperature <= self.minTemp:
                        self.initTemp *= self.tempReheatFactor
                        self.tempStep *= self.tempReheatFactor
                        self.temperature = self.initTemp

                except KeyboardInterrupt:
                    break

        # This only runs when the function completes
        updateProgress()
        updateGraphs(fig, ax)
        updateGraphs(fig, ax, save=True)

        # print("\n", frame, 'positions visited,', self.stats['comparisonsMade'], 'comparisons made')

        save_object(
            {"mockupImg": self.mockupImg, "comboGrid": self.comboGrid},
            f"{self.basePath}results/resume.pkl",
        )
