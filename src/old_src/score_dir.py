import cv2
from sys import argv
from skimage.measure import compare_ssim, compare_mse, compare_nrmse, compare_psnr
import os

image1 = cv2.imread(argv[1], cv2.IMREAD_GRAYSCALE)
sortby = argv[3] if len(argv) > 3 else "psnr"

scores = {}
for fn in os.listdir(argv[2]):
    image2 = cv2.imread(os.path.join(argv[2], fn), cv2.IMREAD_GRAYSCALE)
    scores[fn] = {
        "psnr": compare_psnr(image1, image2),
        "ssim": compare_ssim(image1, image2),
    }

for fn in sorted(scores, key=lambda x: scores[x][sortby], reverse=True):
    # print(fn)
    # print(f"PSNR: {scores[fn]['psnr']:5.3f} SSIM: {scores[fn]['ssim']:5.3f}\n")

# cv2.imwrite('diff.png', cv2.subtract(image1, image2))
