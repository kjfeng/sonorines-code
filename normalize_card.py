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
def read_img(rawpath):
    # img is a 3D numpy.ndarray: h x w x 3channels
    img = cv2.imread(rawpath)
    return img

# converts a 3-channel rgb image to one-channel bw image, and then duplicates gray color to all 3 channels
# for use on the blank card images. Accepted argument is a RGB matrix
def convert_to_bw(imageRGB):
    imgBW = cv2.cvtColor(imageRGB, cv2.COLOR_BGR2GRAY)
    imageFullBW = np.zeros(shape=imageRGB.shape)
    height = imageRGB.shape[0]
    width = imageRGB.shape[1]
    for y in range(height):
        for x in range(width):
            grayVal = imgBW[y][x]
            imageFullBW[y][x] = [gravVal, gravVal, grayVal]
    return imgFullBW

# blurs image (also for use on blank card image)
def blur(image):
    return

# divides card with blank sheet of paper and returns new image
def normalize_pair(cardStringPath, blankStringPath):
    # cardPath and blankPath are strings
    cardPath = r'{}'.format(cardStringPath)
    blankPath = r'{}'.format(blankStringPath)

    # outputs images as h x w x 3channels matrices
    # shape outputs a tuple (h, w, c)
    card = cv2.imread(cardPath)
    blank = cv2.imread(blankPath)

    if card == None:
        print("IOError: sonorine image path cannot be opened", file=stderr)
    if blank == None:
        print("IOError: blank card image path cannot be opened", file=stderr)

    if card.height != blank.height or card.width != blank.width:
        print("The two images need to be of the same size", file=stderr)

    # convert images to RGB if not already

    width, height = card.size
    newImg = Image.new(mode="RGB", size=(width, height))

    # sum values and add to list for both card and blank images
    # pop max from both lists and compare
    # if card's is larger, then divide by max(blank) to get the ratio
    # divide every r g b in card by this value
    # then divide every r g b in card with r g b in blank and return as new image

    cardSumChamp = 0
    blankSumChamp = 0
    ratioChamp = 0
    cardRGB = np.zeros(shape=(width*height, 3))
    blankRGB = np.zeros(shape=(width*height, 3))

    for y in range(height): # y is row
        for x in range(width): # x is col
            # mapping 2D to 1D in index
            linearIndex = y*width + x

            rBlank, gBlank, bBlank = blank.getpixel((x, y))
            blankRGB[linearIndex] = [rBlank, gBlank, bBlank]
            sumBlank = rBlank + gBlank + bBlank

            rCard, gCard, bCard = card.getpixel((x, y))
            if sumBlank == 0:
                cardRGB[linearIndex] = [0, 0, 0]
            else:
                cardRGB[linearIndex] = [rCard, gCard, bCard]
            sumCard = rCard + gCard + bCard

            pixelDiv = [rCard/rBlank, gCard/gBlank, bCard/bBlank]
            maxPixelRatio = max(pixelDiv)
            if maxPixelRatio > 1:
                for div in pixelDiv:
                    div /= maxPixelRatio

            # if sumCard > cardSumChamp:
            #     cardSumChamp = sumCard

            # if sum of blanks are 0, then set card's as 0 and ratio to 1
            # otherwise, get the ratio and compare w champ



            if sumBlank == 0:


            # if sumBlank > blankSumChamp:
            #     blankSumChamp = sumBlank

    if cardSumChamp > blankSumChamp:
        ratio = cardSumChamp / blankSumChamp
        cardRGBNormalized = cardRGB / ratio

    newImgRGB = cardRGBNormalized / blankRGB
    for y in range(height):
        for x in range(width):
            linearIndex = y*width + x
            newR = newImgRGB[linearIndex][0] * 255
            newG = newImgRGB[linearIndex][1] * 255
            newB = newImgRGB[linearIndex][2] * 255
            print(newR)
            # rounding errors might occur here
            newImg.putpixel((x, y), (newR, newG, newB))


    return newImg

# scales single card upon determining the max calue across all channels and cards
def scale_card(maxVal, image):
    return

# creates and returns new image
def create_normalized_img(normalizedMatrix):



def main(argv):
    args = argv[1:]
    if len(args) != 8:
        print("Usage: image_analyzer.py [4 pairs of sonorine<->blank card images]", file=stderr)
        exit(1)
    # make array of raw string paths
    paths = []
    for arg in args:
        paths.append(r'{}'.format(arg))
    sonoImgs = []
    blankImgs = []
    for i in range(len(paths)):
        img = cv2.imread(paths[i])
        # evens = sonorine images
        if i % 2 == 0:
            sonoImgs.append(img)
        else:
            blankImgs.append(img)

    # test stmt
    if len(sonoImgs) != len(blankImgs):
        print('lengths of sonorines array and blank array are different', file=stderr)
        exit(1)








# ----------------------------------------------------------------------------------

if __name__ == '__main__':
    main(argv)
