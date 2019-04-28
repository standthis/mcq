#!/usr/bin/python

'''
#logging.debug('This is a debug message')
logging.info('This is an info message')
logging.warning('This is a warning message')
logging.error('This is an error message')
logging.critical('This is a critical message')
'''

# Python 2/3 compatibility
from __future__ import print_function

from deskew import deskew
import logging
import numpy as np
import cv2 as cv
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import imutils.convenience as conv
import imutils.contours as cont
import sys
import collections 
import math
import random as rng
np.set_printoptions(threshold=sys.maxsize)
logging.basicConfig(level=logging.DEBUG)

def main():
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 'extra/clean.png'
        fn = 'template.png'
    
    memo_path = 'extra/burst/pg_0003-1.png'
    orig = 'template_color.png'
    fnimg = cv.imread(cv.samples.findFile(fn))
    memo_img = cv.imread(cv.samples.findFile(memo_path))
    origimg = cv.imread(cv.samples.findFile(orig))
    fnimg_orig = fnimg.copy()
    fnimg_c = grayed(fnimg)
    origimg_c = grayed(origimg)
    memo_img_c = grayed(memo_img)

    fnimg = sifted(fnimg_c.copy(), origimg_c.copy())
    memo_sift = sifted(memo_img_c.copy(), origimg_c.copy())

    outimg, quiz = findcircles(origimg_c.copy())
    targetout, targetQuiz = findcircles(fnimg.copy())

    studentnum_circles, tasknum = studentnumCircle(origimg_c.copy())
    studentout, studentnum_result = extract(studentnum_circles, fnimg.copy())
    taskout, tasknum_result = extract(tasknum, fnimg.copy())
    tasknumstr = getTasknum(tasknum_result)
    #assert onlyone(studentnum_result)
    if not onlyone(studentnum_result):
        logging.debug("ONLYONE FAIL!")
    #print(studentnum_result)

    studentnum = findStudentnum(studentnum_result)
    #logging.debug(studentnum)

    blankout, answers = extract(quiz, fnimg.copy())
    outmemo, solutions = extract(quiz, memo_sift.copy())

    assert onlyone(solutions)

    result = mark(answers, solutions)
    print('the student recieved', sum(result)/len(result)*100, '%')

    # mark to csv 
    csvl = marktocsv(answers)
    #logging.debug(csvl)
    #logging.debug(len(csvl))
    writecsv(csvl, studentnum , tasknumstr)

    blankout = fnimg
    #blankout = targetout 
    if outimg.any() == None:
        print("outimg is none")
    #cv.imshow("source1", blankout)
    #cv.imshow("source2", cimg)
    k = cv.waitKey(0)
    if k == ord('q'):
        cv.destroyAllWindows()
    #if k == ord('s'):
    #    cv.imwrite("research.png", blankout)
    #cv.imwrite("research.png", blankout)
    print('Done')

def writecsv(mark, stnum, tasknum):
    f = open("results.txt","a")
    letters = None
    for l, line in enumerate(mark):
        if line is not None:
            letters = ''.join(line)
        out = str(stnum) + ", " + str(tasknum) + ", " + str(l) + ", "  + str(letters)
        f.write(out + '\n')
    f.close()

def getTasknum(result):
    #assert len(result) == 2
    output = ""
    if sum(result[0]) == 0:
        output += '0'
    else: 
        output += str(result[0].index(True))
    if sum(result[1]) == 0:
        return None
    else:
        output += str(result[1].index(True))
    return output

def findStudentnum(result):
    student = [] 
    for i, r in enumerate(result):
        if sum(r) == 1:
            if i == 2:
                student.append(chr(97 + r.index(True)))
            else:
                student.append(str(r.index(True)))
        else:
            return None
    return ''.join(student)



def show_wait_destroy(img, winname='win'):
    cv.imshow(winname, img)
    cv.moveWindow(winname, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)

def sifted(img1, img2):
    img1orig = img1.copy()
    img2orig = img2.copy()
    sift = cv.xfeatures2d.SIFT_create()
    orb = cv.ORB_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    #kp1, des1 = orb.detectAndCompute(img1, None)
    #kp2, des2 = orb.detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)
# store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 1
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img2.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.warpPerspective(img1, M, (w,h))
        return dst

def onlyone(result):
    for q in result:
        if sum(q) > 1:
            return False
    return True

def flat(l):
    return [item for sublist in l for item in sublist]

# find answers
def mark(answers, solutions):
    return [a == b for a,b in zip(answers, solutions)]

def marktocsv(answers):
    options = ['a', 'b', 'c', 'd', 'e']
    mark = []
    for qi, q in enumerate(answers):
        letters = []
        for ci, circle in enumerate(q):
            if circle:
                letters.append(options[ci])
        if sum(q) == 0:
            mark.append(None)
        else:
            mark.append(letters)

    return mark


def extract(quiz, warped):
    blank = np.zeros(warped.shape, dtype="uint8")
    answers = []
    for q in quiz:
        answer = []
        for i in q:
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
            total = cv.countNonZero(result)
            h, w = result.shape
            area = h * w 
            answer.append((total / area) > 0.5)
        answers.append(answer)
    #answers = nest(answers, 5)
    return blank, answers

def nest(simple, n):
    return [simple[i:i + n] for i in range(0, len(simple), n)]

def getHough(img, studentnum):
    if studentnum:
        circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,15,param1=10,param2=10,minRadius=10,maxRadius=10)
    else:
        circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,15,param1=10,param2=10,minRadius=7,maxRadius=9)
    circles = np.uint16(np.around(circles))
    return circles

def studentnumCircle(cimg):
    # filter student number
    student = getHough(cimg.copy(), True)
    student = [circ for circ in student[0] if circ[1] < 910]
    student = [circ for circ in student if circ[0] < 280]
    # isolate task
    tasknum = [circ for circ in student if (circ[0] > 170 and circ[1] > 470)]
    # remove task
    student = [circ for circ in student if not (circ[0] > 170 and circ[1] > 470)]
    start = 48
    studentcols = []
    for i in range(7):
        studentcols.append(
                [circ for circ in student 
                    if circ[0] >= start and circ[0] < start + 30])
        start += 30
        #i += 1

    studentcols = [sorted(col, key=lambda x: x[1]) for col in studentcols]

    # Task number sort
    # sort x axis
    tasknum.sort(key=lambda x: x[0])
    # sort y axis
    tasknum = nest(tasknum, 10)
    tasknum = [sorted(col, key=lambda x: x[1]) for col in tasknum]
    return studentcols, tasknum

def findcircles(cimg):
    circles = getHough(cimg.copy(), False)
    circles_orig = circles.copy()
    circles = [circ for circ in circles[0] if circ[0] > 340]


    # filter questions circles
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
    return cimg, quiz

def grayed(src):
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    return gray

if __name__ == '__main__':
    #print(__doc__)
    main()
    #cv.destroyAllWindows()
