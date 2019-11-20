#!/usr/bin/env python

# *****************************************************************************
#  normalize_card.py
#  Usage:
#  Author: Kevin Feng
#  Description:
#   normalizes sonorine images against images of blank cards and outputs 8-bit png
# *****************************************************************************

import cv2
import math
import numpy as np
from sys import argv, stderr, exit
import time
from create_hist import create_hist_from_img


# converts a 3-channel rgb image to one-channel bw image, and then duplicates gray color to all 3 channels
# for use on the blank card images. Accepted argument is a RGB matrix
def convert_to_bw(imageRGB):
    imgBW = cv2.cvtColor(imageRGB, cv2.COLOR_BGR2GRAY)
    imgFullBW = np.repeat(imgBW[:, :, np.newaxis], 3, axis=2)
    return imgFullBW


# divides card with blank sheet of paper and returns normalized image
def normalize_pair(sonoImg, blankImg):
    return (sonoImg / blankImg) * 255


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

def normalize_all(paths):
    sonoImgs = []
    blankImgs = []
    maxChamp = 0
    for i in range(len(paths)):
        img = cv2.imread(paths[i])
        if img.any() == None:
            print('image', i, 'cannot be read!', file=stderr)
            exit(1)

        maxVal = np.max(img)
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
        normalized = normalize_pair(sonoImgs[i], blankBlurred) / maxChamp
        normalized *= 255
        normalized = normalized.astype('float32')
        normBW = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)

        # one-channel bw images (2D arrays)
        finalImgs.append(normBW)

    return finalImgs

def scale_to_pos(img):
    newImg = img
    min = np.min(newImg)
    if min < 0:
        deficit = 0 - min
        newImg += deficit
    # scales entire img to 0-255
    max = np.max(newImg)
    ratio = 255 / max
    newImg /= ratio

    return newImg

def visulize_by_HSV(finalImgs):
    if len(finalImgs) != 4:
        print("Error:", len(finalImgs), "images were normalized, not 4.", file=stderr)
    imgNW = finalImgs[0]
    imgNE = finalImgs[1]
    imgSW = finalImgs[2]
    imgSE = finalImgs[3]

    height = imgNW.shape[0]
    width = imgNW.shape[1]

    da = imgNW - imgSE
    db = imgNE - imgSW
    magnitudes = np.sqrt(da*da + db*db)

    maxDa = np.amax(imgNW, axis=2) - np.amin(imgSE, axis=2)
    maxDb = np.amax(imgNE, axis=2) - np.amin(imgSW, axis=2)
    maxMagnitude = np.sqrt(maxDa*maxDa + maxDb*maxDb)

    # a_scaled = scale_to_pos(da)
    # b_scaled = scale_to_pos(db)

    x1 = 1
    y1 = 0
    x2 = da
    y2 = db

    dot = x1*x2 + y1*y2
    det = x1*y2 - y1*x2
    angleRad = math.atan2(det, dot)

    if angleRad < 0:
        diff = 2 * math.pi + angleRad
        angleRad = diff

    angleDeg = angleRad * 180 / math.pi

    h = angle * 360/179
    s = magnitudes * 255/maxMagnitude
    v = np.full((height, width), 0.75 * 255)

    # create new h x w ndarray, convert to hsv colourspace, insert hsv??
    newImg = np.zeros((height, width, 3))
    hsvImg = cv2.cvtColor(newImg, cv2.COLOR_BGR2HSV)
    hsvImg[:, :, 0] += h
    hsvImg[:, :, 1] += s
    hsvImg[:, :, 2] += v

    return hsvImg


def main(argv):
    args = argv[1:]
    # if len(args) != 8:
    #     print("Usage: image_analyzer.py [4 pairs of sonorine<->blank card images]", file=stderr)
    #     exit(1)
    # make array of raw string paths
    paths = []

    startTime = time.time()

    for arg in args:
        paths.append(r'{}'.format(arg))
    finalImgs = normalize_all(paths)


    for i in range(len(finalImgs)):
        # VARIABLE DEPENDING ON USER'S COMPUTER
        fileString = '/Users/feng/Documents/Kevin/Pton/Classes/cos-iw/sonorines-code/images/normalized' + str(i) + '.png'
        status = cv2.imwrite(fileString, finalImgs[i])
        print('Normalized image ' + str(i) + ' written to file', status)

    endTime = time.time()

    print('Done! Time:', round(endTime - startTime, 2), 's')



# ----------------------------------------------------------------------------------

if __name__ == '__main__':
    main(argv)
