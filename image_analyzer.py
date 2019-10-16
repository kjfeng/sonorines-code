# *****************************************************************************
#  image_analyzer.py
#  Usage: image_analyzer.py [sonorine_scan_path] [blank_card_path]
#  Author: Kevin Feng
#  Description:
#   functions that work on analyzing the sonorine scans. RGB images only!
# *****************************************************************************

import cv2
import math
import numpy as np
from sys import argv, stderr, exit
from PIL import Image

# divides card with blank sheet of paper and returns new image
def normalize_card(cardStringPath, blankStringPath):
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
    cardRGB = np.zeros(shape=(width*height, 3))
    blankRGB = np.zeros(shape=(width*height, 3))

    for i in range(height):
        for j in range(width):
            # mapping 2D to 1D in index
            linearIndex = i*width + j
            rCard, gCard, bCard = card.getpixel(width, height)
            cardRGB[linearIndex] = [rCard, gCard, bCard]
            sumCard = rCard + gCard + bCard
            if sumCard > cardSumChamp:
                cardSumChamp = sumCard

            rBlank, gBlank, bBlank = blank.getpixel(width, height)
            blankRGB[linearIndex] = [rBlank, gBlank, bBlank]
            sumBlank = rBlank + gBlank + bBlank
            if sumBlank > blankSumChamp:
                blankSumChamp = sumBlank

    if cardSumChamp > blankSumChamp:
        ratio = cardSumChamp / blankSumChamp
        cardRGBNormalized = cardRGB / ratio

    newImgRGB = cardRGBNormalized / blankRGB
    for i in range(height):
        for j in range(width):
            linearIndex = i*width + j
            newR = newImgRGB[linearIndex][0]
            newG = newImgRGB[linearIndex][1]
            newB = newImgRGB[linearIndex][2]
            newImg.putpixel((i, j), (newR, newG, newB))


    return newImg


def main(argv):
    args = argv[1:]
    if len(args) != 2:
        print("Usage: image_analyzer.py [sonorine_scan_path] [blank_card_path]", file=stderr)
        exit(1)

    normalizedImg = normalize_card(args[0], args[1])
    print('Normalized')
    normalizedImg.show()


# ----------------------------------------------------------------------------------

if __name__ == '__main__':
    main(argv)
