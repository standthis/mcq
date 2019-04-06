#!/usr/bin/python
import cv2 as cv
import numpy as np

img = cv.imread('extra/mcq.pdf')


cv.imshow("Black", img)
cv.waitKey(0)
cv.destroyAllWindows()

#cv.cvtColor(image, cv2.COLOR_BGR2GRAY)
