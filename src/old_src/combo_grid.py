import numpy as np
from combo import Combo


class ComboGrid:
    # A grid of shape (rows, cols) of Combos.
    # 1 is the space (empty) character. It differs from having no constraint.
    def __init__(self, rows, cols, random=0):
        # print("ComboGrid init: ", rows, cols)
        self.rows = rows
        self.cols = cols
        # Intiialize grid to space character
        self.grid = np.array(
            [[[1, 1, 1, 1] for _ in range(cols)] for _ in range(rows)], dtype=object
        )
        # Order of self.dirty is TL, TR, BL, BR
        self.maxFlips = 2
        self.initDirty()

    def initDirty(self):
        self.dirty = np.array(
            [[[1, 1, 1, 1] for _ in range(self.cols)] for _ in range(self.rows)],
            dtype=bool,
        )
        # Space character on edges never changes, so clear dirty bit
        self.dirty[0, :, :2] = 0
        self.dirty[-1, :, 2:] = 0
        self.dirty[:, 0, 0] = 0
        self.dirty[:, 0, 2] = 0
        self.dirty[:, -1, 1] = 0
        self.dirty[:, -1, 3] = 0
        self.flips = np.array(
            [[0 for _ in range(self.cols)] for _ in range(self.rows)], dtype="object"
        )
        # self.flips[0,:] = 255
        self.flips[-1, :] = 255
        # self.flips[:,0] = 255
        self.flips[:, -1] = 255

    # Put charID at the bottom right of (row,col), bottom left of col+1, etc
    def put(self, row, col, charID, chosen=False):
        # if not self.isDirty(row, col):
        #     print("error! putting char at clean pos")
        #     exit()
        self.grid[row, col, 3] = charID
        self.grid[row + 1, col, 1] = charID
        self.grid[row, col + 1, 2] = charID
        self.grid[row + 1, col + 1, 0] = charID
        # if chosen:
        # self.flips[row, col] += 1
        # print(self.flips[row, col])
        # if chosen:
        #     self.setDirty(row, col, False)
        # else:
        self.setDirty(row, col)

    def setDirty(self, row, col, isDirty=True):
        # Set dirty bits
        self.dirty[row, col, 3] = isDirty
        self.dirty[row + 1, col, 1] = isDirty
        self.dirty[row, col + 1, 2] = isDirty
        self.dirty[row + 1, col + 1, 0] = isDirty

    def isDirty(self, row, col):
        dirty = np.any(self.dirty[row : row + 2, col : col + 2])
        # print(row, col, 'is', dirty)
        # return dirty and self.flips[row, col] <= self.maxFlips
        return dirty

    def isDitherDirty(self, row, col):
        dirty = np.any(
            self.dirty[
                max(0, row - 2) : min(self.rows, row + 4),
                max(0, col - 2) : min(self.cols, col + 4),
            ]
        )
        # print(row, col, 'is', dirty)
        return dirty and self.flips[row, col] <= self.maxFlips

    def clean(self, row, col):
        # print("Cleaning position", row, col)
        self.setDirty(row, col, False)

    def get(self, row, col):
        return self.grid[row, col]

    def getLayers(self):
        layers = []
        origGrid = self.grid.copy()
        for i in range(4):
            skipEvenRow = True
            skipEvenCol = True
            if i == 1:
                skipEvenCol = False
            elif i == 2:
                skipEvenRow = False
            elif i == 3:
                skipEvenCol = False
                skipEvenRow = False
            self.grid = np.array(
                [[[1, 1, 1, 1] for _ in range(self.cols)] for _ in range(self.rows)],
                dtype=object,
            )
            for j, row in enumerate(origGrid[:, :, 3]):
                if (j % 2 == 0 and skipEvenRow) or (j % 2 != 0 and not skipEvenRow):
                    continue
                for k, char in enumerate(row):
                    if (k % 2 == 0 and skipEvenCol) or (k % 2 != 0 and not skipEvenCol):
                        continue
                    if j < self.rows - 1 and k < self.cols - 1:
                        self.put(j, k, char)
            layers.append(self.grid.copy())
        return layers

    def getPrintable(self):
        return np.array([[cell[0] for cell in row] for row in self.grid], dtype=object)

    def __str__(self):
        s = "   "
        for col in range(self.grid.shape[1]):
            s += f"    {col:2} "
        divider = " " + "-" * (len(s))
        s += "\n" + divider + "\n"
        for row in range(self.grid.shape[0]):
            s1 = f" {row:2} | "
            for col in range(self.grid.shape[1]):
                s1 += (
                    f"{self.grid[row, col][0] or 0:2} {self.grid[row, col][1] or 0:2}  "
                )
            s2 = "    | "
            for col in range(self.grid.shape[1]):
                s2 += (
                    f"{self.grid[row, col][2] or 0:2} {self.grid[row, col][3] or 0:2}  "
                )
            s += s1 + "\n" + s2 + "\n\n"
        return s

    def printDirty(self):
        s = "   "
        for col in range(self.dirty.shape[1]):
            s += f"    {col:2} "
        divider = " " + "-" * (len(s))
        s += "\n" + divider + "\n"
        for row in range(self.dirty.shape[0]):
            s1 = f" {row:2} | "
            for col in range(self.dirty.shape[1]):
                s1 += f"{self.dirty[row, col, 0] or 0:2} {self.dirty[row, col, 1] or 0:2}  "
            s2 = "    | "
            for col in range(self.dirty.shape[1]):
                s2 += f"{self.dirty[row, col, 2] or 0:2} {self.dirty[row, col, 3] or 0:2}  "
            s += s1 + "\n" + s2 + "\n\n"
        # print(s)
