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
import csv

def write():
    with open('info.csv', mode='a') as info_file:
        info_writer = csv.writer(info_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        info_writer.writerow((100, 150, 50))
        info_writer.writerow((300, 301))
        info_writer.writerow((101, 150, 49))
        info_writer.writerow((200, 201))

def read():
    with open('info.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count % 2 == 0:
                print(f'Odd Column', row)
                print('cx:', row[0], 'cy:', row[1], 'r:', row[2])
                line_count += 1
            else:
                print(f'Even Column', row)
                print('hx:', row[0], 'hy:', row[1])
                line_count += 1
    print(f'Processed {line_count} lines.')


def main(argv):
    # write()
    read()



# ----------------------------------------------------------------------------------

if __name__ == '__main__':
    main(argv)
