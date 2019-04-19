#!/usr/bin/python
import cv2 as cv
import sys
import numpy as np
from matplotlib import pyplot as plt
import imutils 
# CHECK OUT IMUTILS!
img = cv.imread('extra/mcq.png')
orig = img.copy()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

binary = cv.bitwise_not(gray)

# morphological operations
kernel = np.ones((5,5),np.uint8)
dilation = cv.dilate(binary,kernel,iterations = 1)
erosion = cv.erode(dilation,kernel,iterations = 1)
dilation = cv.dilate(erosion,kernel,iterations = 1)
erosion = cv.erode(dilation,kernel,iterations = 1)
dilation = cv.dilate(erosion,kernel,iterations = 1)
closing = cv.morphologyEx(dilation, cv.MORPH_CLOSE, kernel)
dilation = cv.dilate(closing,kernel,iterations = 1)
gradient = cv.morphologyEx(binary, cv.MORPH_GRADIENT, kernel)

(cnts, hirarchy) = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
convex = [cnt for cnt in cnts if cv.isContourConvex(cnt)]
x, y, w ,h = cv.boundingRect(cnts[0])
for c in cnts:
    print(cv.contourArea(c))

# HOUGH CIRCLES

#img = cv.imread('opencv-logo-white.png',0)
#img = cv.medianBlur(img,5)
#blur = cv.medianBlur(img,5)
#cimg = cv.cvtColor(gray,cv.COLOR_GRAY2BGR)
cimg = gray
#circles = cv.HoughCircles(cimg,cv.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
circles = cv.HoughCircles(cimg,cv.HOUGH_GRADIENT, 5, 500)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
cv.imshow('detected circles',cimg)
k = cv.waitKey(0)
if k == ord('q'):
    cv.imwrite("circle.png", cimg)
cv.destroyAllWindows()
sys.exit()


##################

#cv.imshow("Black", dilation)
#k = cv.waitKey(0)
#if k == ord('q'):
#    cv.imwrite("test.png", dilation)
#cv.destroyAllWindows()
#sys.exit()

#x, y = [], []


#for contour_line in cnts:
#    for contour in contour_line:
#        x.append(contour[0][0])
#        y.append(contour[0][1])


x1 = x
x2 = x + w 
y1 = y
y2 = y + h

print(x1)
print(x2)
print(y2)
print(y1)
#sys.exit()
#x1 = 25
#x2 = 850
#y1 = 50
#y2 = 1225
#print(cnts)
#cnts = cnts[0] if imutils.is_cv2() else cnts[1]  
#cntsSorted = sorted(cnts, key=lambda x: cv.contourArea(x))
cropped = binary[y1:y2, x1:x2]

cv.imshow("Black", cropped)
k = cv.waitKey(0)
if k == ord('q'):
    cv.imwrite("test1.png", cropped)
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
