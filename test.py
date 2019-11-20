# *****************************************************************************
#  test.py
#  Author: Kevin Feng
#  Description:
#   playground
# *****************************************************************************

import cv2
import math
import numpy as np
from sys import argv, stderr, exit
import time
from create_hist import create_hist_from_img


def main(argv):
    args = argv[1:]
    rawpath = r'{}'.format(args[0])
    img = cv2.imread(rawpath, 0)

    # find histogram using numpy.ravel()
    create_hist_from_img(img)



# ----------------------------------------------------------------------------------

if __name__ == '__main__':
    main(argv)
