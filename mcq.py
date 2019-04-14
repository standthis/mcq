#!/usr/bin/python

'''
This example illustrates how to use cv.HoughCircles() function.

Usage:
    mcq.py [<image_name>]
    image argument defaults to extra/clean.png
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import imutils.convenience as conv
import imutils.contours as cont
import sys
import collections 
np.set_printoptions(threshold=sys.maxsize)

def main():
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 'extra/clean.png'
    
    orig = 'extra/clean.png'
    paper, warped = transform(orig)
    _, target = transform(fn)
    outimg, quiz = findcircles(warped.copy())
    outimg, answers = extract(quiz, target)
    if outimg.any() == None:
        print("outimg is none")
        sys.exit()
    cv.imshow("source1", outimg)
    #cv.imshow("source2", cimg)
    k = cv.waitKey(0)
    if k == ord('s'):
        cv.imwrite("research.png", outimg)
    print('Done')



# REMEMBER YOU KNOW WHERE ALL THE CIRCLES ARE FROM THE FIRST PNG
# THIS MEANS YOU CAN OVERLAY TO CHECK QUESTIONS AND POSITIONS
# THIS ALSO MEANS ORIENTATION
# IT MEANS THAT LOST CIRCLES CAN BE RECOVERED FROM THE PROTOTYPE

# find answers
def marking(answers):
    options = ['a', 'b', 'c', 'd', 'e']
    mark = []
    for qi, q in enumerate(answers):
        letters = []
        for ci, circle in enumerate(q):
            if circle:
                letters.append(options[ci])
        mark.append(letters)

    return mark

def extract(quiz, warped):
    blank = np.zeros(warped.shape, dtype="uint8")
    answers = []
    ccc = 0
    for q in quiz:
        for i in q:
            # draw the outer circle
            y = i[0]
            x = i[1]
            r = i[2] + 1
            roi = warped[x-r:x+r, y-r:y+r]
            result = warped[x-r:x+r, y-r:y+r]
            ret, mask = cv.threshold(result, 10, 255, cv.THRESH_BINARY)
            mask_inv = cv.bitwise_not(mask)
            img1_bg = cv.bitwise_and(roi,roi,mask = mask_inv)
            img2_fg = cv.bitwise_and(result, result,mask = mask)
            img2_fg = cv.bitwise_not(img2_fg)
            ret, img2_fg = cv.threshold(img2_fg, 100, 255, cv.THRESH_BINARY)
            result = img2_fg
            dst = cv.add(img1_bg,img2_fg)
            blank[x-r:x+r, y-r:y+r] = dst
            #result = cv.bitwise_not(result)
            #thresh = cv.threshold(result, 0, 255,
            #        cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
            #mask = np.zeros(thresh.shape, dtype="uint8")
            #mask = cv.bitwise_and(, result, mask=mask)
            total = cv.countNonZero(result)
            answers.append(total > 100)
            print(ccc, total)
            if ccc == 10000:
                return blank
            ccc += 1
            #result = img[a1-r:a1+r,b1-r:b1+r]
            #cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            #cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    # two ways 
    # one 
    # find all marked circles and check their positions
    # two
    # cut each circle out and check if marked
    answers = nest(answers, 5)
    return blank, answers

def nest(simple, n):
    return [first[i:i + n] for i in range(0, len(first), n)]

def findcircles(cimg):
    #thresh = cv.threshold(cimg, 0, 255,
    #        cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    #bitwize = cv.bitwise_not(warped)
    #circles = cv.HoughCircles(cimg,cv.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=100)
    circles = cv.HoughCircles(cimg,cv.HOUGH_GRADIENT,1,15,
                                            param1=10,param2=10,minRadius=7,maxRadius=9)
    #circles = cv.HoughCircles(cimg,cv.HOUGH_GRADIENT, 5, 500)
    circles = np.uint16(np.around(circles))
    circles_orig = circles.copy()
    # filter questions circles
    circles = [circ for circ in circles[0] if circ[0] > 340]
    first = [circ for circ in circles if circ[0] < 470]
    second = [circ for circ in circles if circ[0] > 580]
    # unsorted_list.sort(key=lambda x: x[3])
    first.sort(key=lambda x: x[1]) 
    second.sort(key=lambda x: x[1]) 
    # [l[i:i + n] for i in range(0, len(l), n)] 
    n = 5
    # all questions 
    quiz = [first[i:i + n] for i in range(0, len(first), n)]
    quiz.extend([second[i:i + n] for i in range(0, len(second), n)])
    quiz = [sorted(q, key=lambda x: x[0]) for q in quiz]

    for q in quiz:
        for i in q:
            # draw the outer circle
            cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    #count = 0
    #for i in circles[0,:]:
    #    # draw the outer circle
    #    cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    #    # draw the center of the circle
    #    cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    #    if count == -1:
    #        break
    #    count += 1
    return cimg, quiz

def transform(fn):
    src = cv.imread(cv.samples.findFile(fn))
    cimg = src.copy() # numpy function
    #img = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    #img = cv.bilateralFilter(img, 11, 17, 17)
    #img = cv.GaussianBlur(img, (5, 5), 0)
    #img = cv.Canny(img, 75, 200)
    #img = cv.bitwise_not(img)
    #img = cv.medianBlur(img, 5)
    #kernel = np.ones((5,5),np.uint8)
    #img = cv.erode(img,kernel,iterations = 1)
    #img = cv.dilate(img,kernel,iterations = 2)
    #img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    #img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

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
    #threshold
    # apply Otsu's thresholding method to binarize the warped
    # piece of paper
    return paper, warped

if __name__ == '__main__':
    #print(__doc__)
    main()
    cv.destroyAllWindows()
