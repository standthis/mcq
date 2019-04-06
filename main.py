#!/usr/bin/python
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('extra/mcq.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

binary = cv.bitwise_not(gray)
(_,contours) = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
for contour in contours:
    (x,y,w,h) = cv.boundingRect(contour)
    xxx = cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

#Morpholoical transformation
kernel = np.ones((5,5),np.uint8)
erosion = cv.erode(img,kernel,iterations = 1)

cv.imshow("Black", xxx)
#cv.imshow("Black", erosion)
cv.waitKey(0)
cv.destroyAllWindows()

# adaptive thresholding compare
#img = cv.medianBlur(gray,5)
#img = gray
#ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
#th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
#            cv.THRESH_BINARY,11,2)
#th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
#            cv.THRESH_BINARY,11,2)
#titles = ['Original Image', 'Global Thresholding (v = 127)',
#            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
#images = [img, th1, th2, th3]
#for i in range(4):
#    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#    plt.title(titles[i])
#    plt.xticks([]),plt.yticks([])
#plt.show()

# Simple thresholding compare
#img = cv.imread('gradient.png',0)
#ret,thresh1 = cv.threshold(gray,127,255,cv.THRESH_BINARY)
#ret,thresh2 = cv.threshold(gray,127,255,cv.THRESH_BINARY_INV)
#ret,thresh3 = cv.threshold(gray,127,255,cv.THRESH_TRUNC)
#ret,thresh4 = cv.threshold(gray,127,255,cv.THRESH_TOZERO)
#ret,thresh5 = cv.threshold(gray,127,255,cv.THRESH_TOZERO_INV)
#titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
#images = [gray, thresh1, thresh2, thresh3, thresh4, thresh5]
#
#for i in range(6):
#    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
#    plt.title(titles[i])
#    plt.xticks([]),plt.yticks([])
#plt.show()



