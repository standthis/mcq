#!/usr/bin/python

'''
This example illustrates how to use cv.HoughCircles() function.

Usage:
    houghcircles.py [<image_name>]
    image argument defaults to board.jpg
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

import sys

def main():
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 'board.jpg'

    src = cv.imread(cv.samples.findFile(fn))
    img = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    #img = cv.bilateralFilter(img, 11, 17, 17)
    #img = cv.GaussianBlur(img, (5, 5), 0)
    #img = cv.Canny(img, 75, 200)
    img = cv.bitwise_not(img)
    #img = cv.medianBlur(img, 5)
    kernel = np.ones((5,5),np.uint8)
    #img = cv.erode(img,kernel,iterations = 1)
    #img = cv.dilate(img,kernel,iterations = 2)
    #img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    #img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    cimg = src.copy() # numpy function

    #circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1, 30)
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1, 30)

    if circles is not None: # Check if circles have been found and only then iterate over these and add them to the image
        a, b, c = circles.shape
        for i in range(b):
            cv.circle(cimg, (circles[0][i][0], circles[0][i][1]), circles[0][i][2], (0, 0, 255), 3, cv.LINE_AA)
            cv.circle(cimg, (circles[0][i][0], circles[0][i][1]), 2, (0, 255, 0), 3, cv.LINE_AA)  # draw center of circle

        cv.imshow("detected circles", cimg)

    cv.imshow("source", img)
    k = cv.waitKey(0)
    if k == ord('q'):
        cv.imwrite("huffme.png", img)
        cv.imwrite("huffcir.png", cimg)
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
