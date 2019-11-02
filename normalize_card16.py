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
import time
from matplotlib import pyplot as plt


# converts a 3-channel rgb image to one-channel bw image, and then duplicates gray color to all 3 channels
# for use on the blank card images. Accepted argument is a RGB matrix
def convert_to_bw(imageRGB):
    imgBW = cv2.cvtColor(imageRGB, cv2.COLOR_BGR2GRAY)
    imgFullBW = np.repeat(imgBW[:, :, np.newaxis], 3, axis=2)
    return imgFullBW

# blurs image (also for use on blank card image)
# arg is 3channel image matrix
# def blur(image):
#     return cv2.medianBlur(image, 5)


# divides card with blank sheet of paper and returns normalized image
def normalize_pair(sonoImg, blankImg):
    return (sonoImg / blankImg) * 65535
    # return (sonoImg / blankImg) * 255

# converts input image (8-bit) into 16-bit
def convertTo16(img):
    # img16 = np.array(img, dtype=np.uint16)
    img16 = img.astype('uint16')
    img16 *= 256
    return img16

# takes an array of paths of the form [sono_img] [blank_img] ... and returns an array of
# normalized sonorine images
def normalize_all(paths):
    sonoImgs = []
    blankImgs = []
    maxChamp = 0
    for i in range(len(paths)):
        img8 = cv2.imread(paths[i])
        # img = cv2.imread(paths[i])

        if img8.any() == None:
        # if img.any() == None:
            print('image', i, 'cannot be read!', file=stderr)
            exit(1)

        img = convertTo16(img8)

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

    finalImgs = []
    for i in range(len(sonoImgs)):
        blankBW = convert_to_bw(blankImgs[i])
        blankBlurred = cv2.medianBlur(blankBW, 5)
        normalized = normalize_pair(sonoImgs[i], blankBlurred)
        normalized /= maxChamp
        normalized *= 65535
        # normalized *= 255
        # norm32 = normalized.astype('uint16') * 65535
        normRounded = normalized.astype('uint16')
        normBW = cv2.cvtColor(normRounded, cv2.COLOR_BGR2GRAY)
        finalImgs.append(normBW)

    return finalImgs

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

    # array of raw string paths
    paths = []

    startTime = time.time()

    for arg in args:
        paths.append(r'{}'.format(arg))
    finalImgs = normalize_all(paths)

    for i in range(len(finalImgs)):

        # VARIABLE DEPENDING ON USER'S COMPUTER
        fileString = '/Users/feng/Documents/Kevin/Pton/Classes/cos-iw/sonorines-code/images/normalized' + str(i) + '.tiff'
        status = cv2.imwrite(fileString, finalImgs[i])
        print('Normalized image ' + str(i) + ' written to file', status)


    endTime = time.time()

    print('Done! Time:', round(endTime - startTime, 2), 's')



# ----------------------------------------------------------------------------------

if __name__ == '__main__':
    main(argv)
