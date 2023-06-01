import numpy as np
from collections import defaultdict as ddict
import cv2


class Combo:
    def __init__(self, TL, TR, BL, BR, shrink=False):
        self.TL = TL
        self.TR = TR
        self.BL = BL
        self.BR = BR
        self.img = self.genComposite(shrink)
        self.avg = np.average(self.img)
        # self.shrunken = cv2.resize(self.img,
        #     dsize=(self.TL.shrunken.shape[1],self.TL.shrunken.shape[0]),
        #     interpolation=cv2.INTER_AREA
        # )

    # charset is the list of char slices from which combos were generated
    def genComposite(self, shrink):
        def toFloat(img):
            return np.array(img / 255, dtype="float32")

        def getTLq(img):
            return img[: img.shape[0] // 2, : img.shape[1] // 2]

        def getTRq(img):
            return img[: img.shape[0] // 2, img.shape[1] // 2 :]

        def getBLq(img):
            return img[img.shape[0] // 2 :, : img.shape[1] // 2]

        def getBRq(img):
            return img[img.shape[0] // 2 :, img.shape[1] // 2 :]

        # Always composite from full-size image (not shrunken)
        TLimg = self.TL.cropped
        TRimg = self.TR.cropped
        BLimg = self.BL.cropped
        BRimg = self.BR.cropped
        TLc = toFloat(getBRq(TLimg))
        TRc = toFloat(getBLq(TRimg))
        BLc = toFloat(getTRq(BLimg))
        BRc = toFloat(getTLq(BRimg))
        img = TLc * TRc * BLc * BRc
        return img


class ComboSet:
    # Container class with useful methods
    # Stores Combos in 4D sparse array for easy filtering by constraint
    def __init__(self, chars=None):
        self.combos = ddict(lambda: ddict(lambda: ddict(lambda: ddict(None))))
        # self.flat = []
        # if chars:
        #     self.genCombos(chars)

    # def genCombos(self, chars):
    #     for TL in chars:
    #         for TR in chars:
    #             for BL in chars:
    #                 for BR in chars:
    #                     combo = Combo(TL, TR, BL, BR)
    #                     self.combos[TL.id][TR.id][BL.id][BR.id] = combo
    #                     # self.flat.append(combo)

    #     print("Generated", len(chars)**4, "combos.")

    # Takes chars
    def genCombo(self, TL, TR, BL, BR):
        combo = Combo(TL, TR, BL, BR)
        # Don't store combos due to memory limitations
        self.combos[TL.id][TR.id][BL.id][BR.id] = combo
        # self.flat.append(combo)
        return combo

    # Takes char IDs
    def getCombo(self, TLid, TRid, BLid, BRid):
        if BRid in self.combos[TLid][TRid][BLid]:
            # print("NS: combo cache hit")
            return self.combos[TLid][TRid][BLid][BRid]
        else:
            # print("NS: combo cache miss")
            return None
