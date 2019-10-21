# *****************************************************************************
#  image_analyzer.py
#  Usage: image_analyzer.py [sonorine_scan_path] [blank_card_path] x 4 pairs
#  Author: Kevin Feng
#  Description:
#   functions that work on analyzing the sonorine scans. RGB images only!
# *****************************************************************************

import cv2
import math
import numpy as np
from sys import argv, stderr, exit
# from PIL import Image

# returns an RGB matrix of an image given its raw path
# def read_img(rawpath):
#     # img is a 3D numpy.ndarray: h x w x 3channels
#     img = cv2.imread(rawpath)
#     return img

# converts a 3-channel rgb image to one-channel bw image, and then duplicates gray color to all 3 channels
# for use on the blank card images. Accepted argument is a RGB matrix
def convert_to_bw(imageRGB):
    imgBW = cv2.cvtColor(imageRGB, cv2.COLOR_BGR2GRAY)
    imgFullBW = np.zeros(shape=imageRGB.shape, dtype=np.uint8)
    height = imageRGB.shape[0]
    width = imageRGB.shape[1]
    for y in range(height):
        for x in range(width):
            grayVal = imgBW[y][x]
            imgFullBW[y][x] = [grayVal, grayVal, grayVal]
    return imgFullBW

# blurs image (also for use on blank card image)
# arg is 3channel image matrix
def blur(image):
    return cv2.medianBlur(image, 5)

# divides card with blank sheet of paper and returns normalized image
def normalize_pair(sonoImg, blankImg):
    return sonoImg / blankImg

# scales single card upon determining the max calue across all channels and cards
# def scale_card(maxVal, image):
#     return image / maxVal


# takes an image matrix and displays image
def show_image(img):
    screen_res = 1280, 720
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)

    cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('dst_rt', window_width, window_height)

    cv2.imshow('dst_rt', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(argv):
    args = argv[1:]
    # if len(args) != 8:
    #     print("Usage: image_analyzer.py [4 pairs of sonorine<->blank card images]", file=stderr)
    #     exit(1)
    # make array of raw string paths
    paths = []
    for arg in args:
        paths.append(r'{}'.format(arg))
    sonoImgs = []
    blankImgs = []
    maxChamp = 0
    for i in range(len(paths)):
        img = cv2.imread(paths[i])
        print(paths[i])
        if img.any() == None:
            print('image', i, 'cannot be read!', file=stderr)
            exit(1)

        maxVal = img.max()
        if maxVal > maxChamp:
            maxChamp = maxVal
        # evens = sonorine images
        if i % 2 == 0:
            sonoImgs.append(img)
        else:
            blankImgs.append(img)

    # test stmt
    if len(sonoImgs) != len(blankImgs):
        print('lengths of sonorines array and blank array are different', file=stderr)
        exit(1)

    normalizedImgs = []

    for i in range(len(sonoImgs)):
        blankBW = convert_to_bw(blankImgs[i])
        blankBlurred = cv2.medianBlur(blankBW, 5)
        print(blankBlurred[1000][1000])
        normalized = normalize_pair(sonoImgs[i], blankBlurred)
        normalized /= maxVal
        normalizedImgs.append(normalized)
    print(normalizedImgs[0][1000][1200])
    # show_image(normalizedImgs[0])



# ----------------------------------------------------------------------------------

if __name__ == '__main__':
    main(argv)
