# *****************************************************************************
#  hsv_vis.py
#  Author: Kevin Feng
#  Description:
#   takes 4 normalized images and visualizes differences in HSV
# *****************************************************************************

import cv2
import math
import numpy as np
from sys import argv, stderr, exit
import time
from create_hist import create_hist_from_img


def main(argv):
    startTime = time.time()

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

    min0 = np.amin(img2)
    max0 = np.amax(img0)

    min1 = np.amin(img3)
    max1 = np.amax(img1)

    magnitudes = np.sqrt(sub0*sub0 + sub1*sub1)

    maxDa = max0 - min0
    maxDb = max1 - min1
    maxMagnitude = np.sqrt(maxDa*maxDa + maxDb*maxDb)

    x1 = np.full(sub0.shape, 1)
    y1 = np.full(sub0.shape, 0)
    x2 = sub0
    y2 = sub1

    # dot = x1*x2 + y1*y2
    # det = x1*y2 - y1*x2
    angleRad = np.arctan(sub1, sub0)

    if angleRad.min() < 0:
        angleRad -= angleRad.min()  # add the deficit so min is 0

    angleDeg = angleRad * 180 / math.pi

    h = angleDeg * 179/angleDeg.max()
    s = magnitudes * 255/maxMagnitude
    v = np.full(angleDeg.shape, 0.5 * 255)

    newImg = colourImg = np.stack((h, s, v), axis=2)
    hsvImg = cv2.cvtColor(np.uint8(newImg), cv2.COLOR_BGR2HSV)

    fileString = '/Users/feng/Documents/Kevin/Pton/Classes/cos-iw/sonorines-code/images/hsv_vis.png'
    # status = cv2.imwrite(fileString, np.uint8(colourImg))
    status = cv2.imwrite(fileString, hsvImg)
    endTime = time.time()
    print('HSV visualization written:', status)
    print('Time:', round(endTime - startTime, 2), 's')


# ----------------------------------------------------------------------------------

if __name__ == '__main__':
    main(argv)
