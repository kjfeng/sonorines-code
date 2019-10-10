# *****************************************************************************
#  image_analyzer.py
#  Author: Kevin Feng
#  Description:
#   functions that work on analyzing the sonorine scans
# *****************************************************************************

import cv2
import math
import numpy as np
from sys import argv, stderr, exit

# divides card with blank sheet of paper
def normalize_card(cardPath, blankPath):
    # vectors is an array of 3D np vectors


    return medoid


def main(argv):
    # convert string args to floats
    args = [float(arg) for arg in argv[1:]]
    if len(args) % 3 != 0:
        print("Enter coordinates in multiples of 3's to create 3D vector", file=stderr)
        exit(1)

    i = 0

    vectors = []
    while i < len(args):
        vector = np.array([args[i], args[i+1], args[i+2]])
        vectors.append(vector)
        i += 3

    print(getMedoid(vectors))


# ----------------------------------------------------------------------------------

if __name__ == '__main__':
    main(argv)
