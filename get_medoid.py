# *****************************************************************************
#  get_medoid.py
#  Author: Kevin Feng
#  Description:
#   Gets the medoid of an array of 3D vectors given from command line
# *****************************************************************************

import math
import numpy as np
from sys import argv, stderr, exit

def get_medoid(vectors):
    # vectors is an array of 3D np vectors
    if len(vectors) == 0:
        print("No vectors given", file=stderr)
        exit(1)

    vectorsNorm = [vector/np.linalg.norm(vector) for vector in vectors]

    dottedSums = []

    for i in range(len(vectorsNorm)):
        if type(vectorsNorm[i]) is not np.ndarray:
            print("Not a NumPy vector", file=stderr)
            exit(1)
        if vectorsNorm[i].shape[0] != 3:
            print("Not a 3D vector", file=stderr)
            exit(1)

        sum = 0
        for j in range(len(vectorsNorm)):
            sum += np.dot(vectorsNorm[i], vectorsNorm[j])

        # subtract dotted with itself
        sum -= np.linalg.norm(vectorsNorm[i])
        dottedSums.append(sum)

    maxDottedSum = max(dottedSums)
    medoid = vectors[dottedSums.index(maxDottedSum)]

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

    print(get_medoid(vectors))


# ----------------------------------------------------------------------------------

if __name__ == '__main__':
    main(argv)
