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
        normalized = normalize_pair(convert_to_bw(sonoImgs[i]), blankBlurred)
        normed.append(normalized)
        # maxVal = normalized.max()
        # if maxVal > maxChamp:
        #     maxChamp = maxVal
        stackedImgs = np.array(normed)

    # find percentiles
    p4 = np.percentile(stackedImgs, 4)
    p96 = np.percentile(stackedImgs, 96)

    for i in range(len(normed)):
        clipped = np.clip(normed[i], p4, p96)
        min = clipped.min()
        max = clipped.max()
        # get smallest element to 0
        # clipped -= min
        rescaled = (clipped - min) / (max - min) * 65535
        normBW = cv2.cvtColor(np.float32(rescaled), cv2.COLOR_BGR2GRAY)
        finalImgs.append(normBW)

    return finalImgs

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

    maxDa = np.amax(imgNW, axis=1) - np.amin(imgSE, axis=1)
    maxDb = np.amax(imgNE, axis=1) - np.amin(imgSW, axis=1)
    maxMagnitude = np.sqrt(maxDa*maxDa + maxDb*maxDb)

    # a_scaled = scale_to_pos(da)
    # b_scaled = scale_to_pos(db)

    # da shape is (1200, 1200)

    x1 = np.full(da.shape, 1)
    y1 = np.full(da.shape, 0)
    x2 = da
    y2 = db

    dot = x1*x2 + y1*y2
    det = x1*y2 - y1*x2
    angleRad = np.arctan2(det, dot)

    if angleRad.min() < 0:
        angleRad -= angleRad.min()  # add the deficit so min is 0
        # diff = 2 * math.pi + angleRad
        # angleRad = diff

    angleDeg = angleRad * 180 / math.pi

    h = angleDeg * 179/angleDeg.max()
    s = magnitudes * 255/maxMagnitude
    v = np.full((height, width), 0.75 * 255)

    newImg = colourImg = np.stack((h, s, v), axis=2)
    hsvImg = cv2.cvtColor(np.uint8(newImg), cv2.COLOR_BGR2HSV)

    return hsvImg


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

    # hsvImg = visulize_by_HSV(finalImgs)

    for i in range(len(finalImgs)):
        fileString = '/Users/feng/Documents/Kevin/Pton/Classes/cos-iw/sonorines-code/images/normalized' + str(i) + '.tif'
        status = cv2.imwrite(fileString, finalImgs[i])
        print('Normalized image ' + str(i) + ' written to file', status)

    # fileString = '/Users/feng/Documents/Kevin/Pton/Classes/cos-iw/sonorines-code/images/hsv.tif'
    # status = cv2.imwrite(fileString, hsvImg)
    # print('HSV image written to file:', status)


    endTime = time.time()

    print('Done! Time:', round(endTime - startTime, 2), 's')



# ----------------------------------------------------------------------------------

if __name__ == '__main__':
    main(argv)
