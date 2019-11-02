# *****************************************************************************
#  creat_hist.py
#  Author: Kevin Feng
#  Description:
#   outputs histogram for selected image
# *****************************************************************************

import cv2
import math
import numpy as np
from sys import argv, stderr, exit
from matplotlib import pyplot as plt

def create_hist(img):
    plt.hist(img.ravel(),256,[0,256])
    plt.show()


def main(argv):
    args = argv[1:]
    if len(args) != 1:
        print("Usage: python create_hist.py [img_path]", file=stderr)
    rawpath = r'{}'.format(args[0])
    img = cv2.imread(rawpath, 0)

    # find histogram using numpy.ravel()
    create_hist(img)



# ----------------------------------------------------------------------------------

if __name__ == '__main__':
    main(argv)
