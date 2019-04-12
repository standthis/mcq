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
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import imutils.convenience as conv
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

    # find contours in the edge map, then initialize
    # the contour that corresponds to the document
    # load the image, convert it to grayscale, blur it
    # slightly, then find edges
    image = src.copy()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edged = cv.Canny(blurred, 75, 200)
    cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL,
	    cv.CHAIN_APPROX_SIMPLE)
    cnts = conv.grab_contours(cnts)
    docCnt = None
     
    # ensure that at least one contour was found
    if len(cnts) > 0:
	# sort the contours according to their size in
	# descending order
        cnts = sorted(cnts, key=cv.contourArea, reverse=True)
	# loop over the sorted contours
        for c in cnts:
	    # approximate the contour
            peri = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, 0.02 * peri, True)

	    # if our approximated contour has four points,
	    # then we can assume we have found the paper
            if len(approx) == 4:
                docCnt = approx
                break

    # apply a four point perspective transform to both the
    # original image and grayscale image to obtain a top-down
    # birds eye view of the paper
    paper = four_point_transform(image, docCnt.reshape(4, 2))
    warped = four_point_transform(gray, docCnt.reshape(4, 2))

    cv.imshow("source", paper)
    cv.imshow("source", warped)
#    img = paper
#    cimg = warped
    #cv.imshow("source", img)
    k = cv.waitKey(0)
    if k == ord('q'):
        cv.imwrite("huffme.png", img)
        cv.imwrite("huffcir.png", cimg)
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
