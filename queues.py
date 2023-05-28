from pqdict import pqdict
import random


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
