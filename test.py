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
from matplotlib import pyplot as plt


def main(argv):
    args = argv[1:]
    rawpath = r'{}'.format(args[0])
    img = cv2.imread(rawpath, 0)

    # find histogram using numpy.ravel()
    plt.hist(img.ravel(),256,[0,256])
    plt.show()



# ----------------------------------------------------------------------------------

if __name__ == '__main__':
    main(argv)
