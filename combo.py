from collections import defaultdict as ddict


def genComposite(TLimg, TRimg, BLimg, BRimg):
    def getTLq(img):
        return img[: img.shape[0] // 2, : img.shape[1] // 2]

    def getTRq(img):
        return img[: img.shape[0] // 2, img.shape[1] // 2 :]

    def getBLq(img):
        return img[img.shape[0] // 2 :, : img.shape[1] // 2]

    def getBRq(img):
        return img[img.shape[0] // 2 :, img.shape[1] // 2 :]

    # Always composite from full-size image (not shrunken)
    TLc = getBRq(TLimg)
    TRc = getBLq(TRimg)
    BLc = getTRq(BLimg)
    BRc = getTLq(BRimg)
    img = TLc * TRc * BLc * BRc
    return img


class Combo:
    def __init__(self, TL, TR, BL, BR, shrink=False):
        self.TL = TL
        self.TR = TR
        self.BL = BL
        self.BR = BR
        self.img = genComposite(
            TL.croppedFloat,
            TR.croppedFloat,
            BL.croppedFloat,
            BR.croppedFloat,
        )


class ComboSet:
    # Container class with useful methods
    # Stores Combos in 4D sparse array for easy filtering by constraint
    def __init__(self, chars=None):
        self.combos = ddict(lambda: ddict(lambda: ddict(lambda: ddict(None))))

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
