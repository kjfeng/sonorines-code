#!/usr/bin/env python

# *****************************************************************************
#  crop.py
#  Usage: python crop.py [path] [outputName.tif]
#  Author: Kevin Feng
#  Description:
#   crops 16-it tif image to 1200 by 1200 pixel and writes it to cropped.tif
# *****************************************************************************

import cv2
import math
import numpy as np
from sys import argv, stderr, exit
import time


def crop_img(path, filename):
    init_x = 6424
    init_y = 4346
    dx = 1200
    dy = 1200
    img = cv2.imread(path, -1)
    newImg = img[init_y:init_y+dy, init_x:init_x+dx]
    fileString = '/Users/feng/Documents/Kevin/Pton/Classes/cos-iw/sonorines-code/images/' + filename
    cv2.imwrite(fileString, newImg)


def main(argv):
    args = argv[1:]
    if len(args) != 2:
        print('Usage: python crop.py [path] [outputName.tif]', file=stderr)
        exit(1)

    startTime = time.time()

    path = r'{}'.format(args[0])
    crop_img(path, args[1])


    endTime = time.time()

    print('Cropped! Time:', round(endTime - startTime, 2), 's')



# ----------------------------------------------------------------------------------

if __name__ == '__main__':
    main(argv)
