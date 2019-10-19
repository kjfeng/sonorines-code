# *****************************************************************************
#  image_analyzer.py
#  Usage: image_analyzer.py [sonorine_scan_path] [blank_card_path] x 4
#  Author: Kevin Feng
#  Description:
#   functions that work on analyzing the sonorine scans. RGB images only!
# *****************************************************************************

import cv2
import math
import numpy as np
from sys import argv, stderr, exit
# from PIL import Image

# converts a 3-channel rgb image to one-channel bw image
# for use on the blank card images. Arg is RGB matrix
def convert_to_bw(imageRGB):
    return
# blurs image (also for use on blank card image)
def blur(image):
    return

# divides card with blank sheet of paper and returns new image
def normalize_card(cardStringPath0, cardStringPath1, cardStringPath2, cardStringPath3, blankStringPath):
    # cardPath and blankPath are strings
    cardPath = r'{}'.format(cardStringPath)
    blankPath = r'{}'.format(blankStringPath)

    card = Image.open(cardPath)
    blank = Image.open(blankPath)

    if card == None:
        print("IOError: 1st image path cannot be opened", file=stderr)
    if blank == None:
        print("IOError: 2nd image path cannot be opened", file=stderr)

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


def main(argv):
    args = argv[1:]
    if len(args) != 5:
        print("Usage: image_analyzer.py [sonorine_scan_path_0] [sonorine_scan_path_1] [sonorine_scan_path_2] [sonorine_scan_path_3] [blank_card_path]", file=stderr)
        exit(1)

    normalizedImg = normalize_card(args[0], args[1], args[2], args[3], args[4])
    print('Normalized')
    normalizedImg.show()


# ----------------------------------------------------------------------------------

if __name__ == '__main__':
    main(argv)
