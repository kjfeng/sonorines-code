# *****************************************************************************
#  image_analyzer.py
#  Usage: image_analyzer.py [sonorine_scan_path] [blank_card_path]
#  Author: Kevin Feng
#  Description:
#   functions that work on analyzing the sonorine scans
# *****************************************************************************

import cv2
import math
import numpy as np
from sys import argv, stderr, exit

# divides card with blank sheet of paper
def normalize_card(cardStringPath, blankStringPath):
    # cardPath and blankPath are strings
    cardPath = r'{}'.format(cardStringPath)
    blankPath = r'{}'.format(blankStringPath)


    return picture


def main(argv):
    if len(args) != 2:
        print("Usage: image_analyzer.py [sonorine_scan_path] [blank_card_path]", file=stderr)
        exit(1)




# ----------------------------------------------------------------------------------

if __name__ == '__main__':
    main(argv)
