# *****************************************************************************
#  rgv_vis.py
#  Author: Kevin Feng
#  Description:
#   takes 4 normalized b/w images of sonorines and performs RGB visualization
# *****************************************************************************

import cv2
import math
import numpy as np
from sys import argv, stderr, exit
import time
from create_hist import create_hist_from_img


def main(argv):
    args = argv[1:]
    rawpath0 = r'{}'.format(args[0])
    rawpath1 = r'{}'.format(args[1])
    rawpath2 = r'{}'.format(args[2])
    rawpath3 = r'{}'.format(args[3])
    img0 = cv2.imread(rawpath0, -1)
    img1 = cv2.imread(rawpath1, -1)
    img2 = cv2.imread(rawpath2, -1)
    img3 = cv2.imread(rawpath3, -1)

    sub0 = img0 - img2
    sub1 = img1 - img3

    min0 = np.amin(sub0)
    max0 = np.amax(sub0)

    min1 = np.amin(sub1)
    max1 = np.amax(sub1)

    # red = 65535 * (sub0 - min0) / (max0 - min0)
    # blue = 65535 * (sub1 - min1) / (max1 - min1)
    red = 255 * (sub0 - min0)
    blue = 255 * (sub1 - min1)
    green = np.full(np.shape(red), 100)
    print('sub0:', sub0)
    print('min0:', min0)
    print('max0:', max0)

    colourImg = np.stack((blue, green, red), axis=2)

    fileString = '/Users/feng/Documents/Kevin/Pton/Classes/cos-iw/sonorines-code/images/resultColour.png'
    status = cv2.imwrite(fileString, colourImg)
    print(status)


# ----------------------------------------------------------------------------------

if __name__ == '__main__':
    main(argv)
