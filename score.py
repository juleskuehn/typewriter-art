import cv2
from sys import argv
from skimage.measure import compare_ssim, compare_mse, compare_nrmse, compare_psnr

image1 = cv2.imread(argv[1], cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(argv[2], cv2.IMREAD_GRAYSCALE)

print("PSNR:", compare_psnr(image1, image2))
print("SSIM:", compare_ssim(image1, image2))

cv2.imwrite("diff.png", cv2.subtract(image1, image2))
