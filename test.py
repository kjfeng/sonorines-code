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

def show_image(img):
    screen_res = 1280, 720
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)

    cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('dst_rt', window_width, window_height)

    cv2.imshow('dst_rt', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(argv):
    args = argv[1:]
    rawpath = r'{}'.format(args[0])
    img = cv2.imread(rawpath)
    print(type(img[0][0][0]))
    imgBlurred = cv2.medianBlur(img, 5)
    # show_image(imgBlurred)


# ----------------------------------------------------------------------------------

if __name__ == '__main__':
    main(argv)
