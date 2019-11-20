# *****************************************************************************
#  image_analyzer.py
#  Usage: image_analyzer.py [sonorine_scan_path] [blank_card_path] x 4 pairs
#  Author: Kevin Feng
#  Description:
#   normalizes and visualizes 16-bit sonorine scans. RGB images only!
# *****************************************************************************

import cv2
import math
import numpy as np
from sys import argv, stderr, exit
import time


# converts a 3-channel rgb image to one-channel bw image, and then duplicates gray color to all 3 channels
# for use on the blank card images. Accepted argument is a RGB matrix
def convert_to_bw(imageRGB):
    imgBW = cv2.cvtColor(imageRGB, cv2.COLOR_BGR2GRAY)
    imgFullBW = np.repeat(imgBW[:, :, np.newaxis], 3, axis=2)
    return imgFullBW


# divides card with blank sheet of paper and returns normalized image
def normalize_pair(sonoImg, blankImg):
    return (sonoImg / blankImg) * 65535

# takes an array of paths of the form [sono_img] [blank_img] ... and returns an array of
# normalized sonorine images
def normalize_all(paths):
    sonoImgs = []
    blankImgs = []
    # highest r/g/b value across all sono images
    maxChamp = 0
    for i in range(len(paths)):
        img = cv2.imread(paths[i], -1)

        if img.any() == None:
            print('image', i, 'cannot be read!', file=stderr)
            exit(1)


        # evens = sonorine images
        if i % 2 == 0:
            sonoImgs.append(img)
        else:
            blankImgs.append(img)

    # test stmt
    if len(sonoImgs) != len(blankImgs):
        print('lengths of sonorines array and blank array are different', file=stderr)
        exit(1)

    normed = []
    finalImgs = []

    for i in range(len(sonoImgs)):
        blankBW = convert_to_bw(blankImgs[i])
        blankBlurred = cv2.medianBlur(blankBW, 5)
        normalized = normalize_pair(sonoImgs[i], blankBlurred)
        normed.append(normalized)
        maxVal = normalized.max()
        if maxVal > maxChamp:
            maxChamp = maxVal

    for i in range(len(normed)):
        normed[i] /= maxChamp
        normed[i] *= 65535
        # normalized /= maxChamp
        # print(normalized.max())
        # normalized *= 65535
        #normRounded = normalized.astype('float16')
        normBW = cv2.cvtColor(np.uint16(normed[i]), cv2.COLOR_BGR2GRAY)
        finalImgs.append(normBW)

    return finalImgs


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
        fileString = '/Users/feng/Documents/Kevin/Pton/Classes/cos-iw/sonorines-code/images/normalized' + str(i) + '.tif'
        status = cv2.imwrite(fileString, finalImgs[i])
        print('Normalized image ' + str(i) + ' written to file', status)


    endTime = time.time()

    print('Done! Time:', round(endTime - startTime, 2), 's')



# ----------------------------------------------------------------------------------

if __name__ == '__main__':
    main(argv)
