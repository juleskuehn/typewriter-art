from pqdict import pqdict
import random
from generator_utils import scoreAnn, scoreMse, compare


class PositionQueue:
    def __init__(self, generator, maxVisits=5):
        self.queue = pqdict({}, key=lambda x: x[0])
        self.generator = generator
        self.changed = {}
        self.maxVisits = maxVisits
        # print("Initializing priority queue")
        for row in range(generator.rows - 1):
            for col in range(generator.cols - 1):
                position = (row, col)
                self.add(position)
        # self.generator.comboGrid.initChanged()

    def add(self, position):
        row, col = position
        visits = self.generator.comboGrid.flips[row, col]
        # print(position, ":", visits, "visits")
        if visits > self.maxVisits:
            return
        bestScore, bestId = self.score(position)
        self.queue[position] = (bestScore, (position, bestId))

    # Returns a tuple (position, bestId)
    def remove(self):
        # row, col = self.queue.topitem()[0]
        # print(self.queue.topitem()[1])
        # Update priorities of surrounding items
        return self.queue.popitem()[1][1]

    def update(self, row, col):
        if row > 0 and col > 0:
            self.add((row - 1, col - 1))
        if row > 0:
            self.add((row - 1, col))
        if row > 0 and col + 2 < self.generator.cols:
            self.add((row - 1, col + 1))
        if col > 0:
            self.add((row, col - 1))
        if col + 2 < self.generator.cols:
            self.add((row, col + 1))
        if row + 2 < self.generator.rows and col > 0:
            self.add((row + 1, col - 1))
        if row + 2 < self.generator.rows:
            self.add((row + 1, col))
        if row + 2 < self.generator.rows and col + 2 < self.generator.cols:
            self.add((row + 1, col + 1))

    def score(self, position):
        row, col = position
        oldScore = compare(self.generator, row, col)
        newScore, bestId = scoreMse(self.generator, row, col)
        if bestId == self.generator.comboGrid.get(row, col)[3]:
            return newScore - oldScore, None
        else:
            return newScore - oldScore, bestId


class LinearQueue:
    def __init__(self, generator, maxVisits=5, randomOrder=False):
        self.queue = []
        self.generator = generator
        self.maxVisits = maxVisits
        self.randomOrder = randomOrder
        # for row in range(generator.rows - 1):
        #     for col in range(generator.cols - 1):
        #         position = (row, col)
        #         self.add(position)
        self.fill()

    def fill(self):
        layers = [0, 3, 2, 1]
        for layerID in layers:
            startRow = 1 if layerID in [2, 3] else 0
            startCol = 1 if layerID in [1, 3] else 0
            endRow = self.generator.rows - 1
            endCol = self.generator.cols - 1
            for row in range(startRow, endRow, 2):
                for col in range(startCol, endCol, 2):
                    visits = self.generator.comboGrid.flips[row, col]
                    if visits <= self.maxVisits:
                        self.queue.append((row, col))
        if self.randomOrder:
            random.shuffle(self.queue)

    def add(self, position):
        if len(self.queue) == 0:
            self.fill()

    # Returns a tuple (position, bestId)
    def remove(self):
        return self.queue.pop(), None  # bestId not calculated for LinearQueue
