#!/usr/bin/python
import cv2 as cv
import sys
import numpy as np
from matplotlib import pyplot as plt
import imutils 
# CHECK OUT IMUTILS!
img = cv.imread('extra/mcq.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

binary = cv.bitwise_not(gray)

(cnts, hirarchy ) = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

x, y = [], []

for contour_line in cnts:
    for contour in contour_line:
        x.append(contour[0][0])
        y.append(contour[0][1])



x1 = 38
x2 = 832
y1 = 160
y2 = 1220
#print(cnts)
#cnts = cnts[0] if imutils.is_cv2() else cnts[1]  
#cntsSorted = sorted(cnts, key=lambda x: cv.contourArea(x))
cropped = binary[y1:y2, x1:x2]

cv.imshow("Black", cropped)
k = cv.waitKey(0)
if k == ord('q'):
    cv.imwrite("test.png", cropped)
cv.destroyAllWindows()
sys.exit()


print(type(contours[0][0, 0, 0]))
for contour in contours:
    (x,y,w,h) = cv.boundingRect(contour)
    what = cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)



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



