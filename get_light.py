# *****************************************************************************
#  get_light.py
#  Author: Kevin Feng
#  Description:
#   Determines light vector given a csv of mirror ball location and highlight
#   coordinates.
# *****************************************************************************

import math
import numpy as np
import csv
from sys import argv, stderr, exit

# reads csv file with name specified in arg (created by gui) and returns two arrays:
# one with the locations of the 16 mirror ball centers, other with highlight coords
def read_csv_file():
    arr_xyr = []
    arr_hl = []

    with open('info.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        line_count = 0
        for row in csv_reader:
            if line_count % 2 == 0:
                arr_xyr.append(row)
                # print(f'Odd Column', row)
                # print('cx:', row[0], 'cy:', row[1], 'r:', row[2])
                line_count += 1
            else:
                arr_hl.append(row)
                # print(f'Even Column', row)
                # print('hx:', row[0], 'hy:', row[1])
                line_count += 1
    print(f'csv file reader processed {line_count} lines.')
    return arr_xyr, arr_hl

# finds the 16 light directions associated with the 16 balls
def find_light(arr_xyr, arr_hl):
    if len(arr_xyr) != len(arr_hl):
        print('Length of ball and highlight locations are not the same', file=stderr)
        exit(1)
    lightVec = []
    for i in range(len(arr_xyr)):
        cx = arr_xyr[i][0]
        cy = arr_xyr[i][1]
        r = arr_xyr[i][2]
        hx = arr_hl[i][0]
        hy = arr_hl[i][1]

        dx = hx - cx
        dy = hy - cy

        d = math.sqrt(dx*dx + dy*dy)
        theta = math.asin(d/r)
        phi = (math.pi/2) - (2*theta)
        dz = d * math.tan(phi)

        l = np.array([dx, dy, dz])
        l = -l/np.linalg.norm(l)
        lightVec.append(l)

    return lightVec

# returns medoid of a given array
# vectors is an array of 3D np vectors
def get_medoid(vectors):

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
    arr_xyr, arr_hl = read_csv_file()
    light_vectors = find_light(arr_xyr, arr_hl)
    light_medoid = get_medoid(light_vectors)
    print(light_medoid)


# ----------------------------------------------------------------------------------

if __name__ == '__main__':
    main(argv)
