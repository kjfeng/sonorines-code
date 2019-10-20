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

def main(argv):
    args = argv[1:]
    rawpath = r'{}'.format(args[0])
    img = cv2.imread(rawpath)
    print('shape of colour matrix is: ', img.shape)
    img_bw = cv2.imread(rawpath, 0)
    print('shape of bw matrix is: ', img_bw.shape)


# ----------------------------------------------------------------------------------

if __name__ == '__main__':
    main(argv)
