# find contours in the thresholded image, then initialize
# the list of contours that correspond to questions
cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE)
cnts = conv.grab_contours(cnts)
questionCnts = []
 
# loop over the contours
print(len(cnts))
for c in cnts:
    # compute the bounding box of the contour, then use the
    # bounding box to derive the aspect ratio
    (x, y, w, h) = cv.boundingRect(c)
    #print(cv.boundingRect(c))
    ar = w / float(h)
    # in order to label the contour as a question, region
    # should be sufficiently wide, sufficiently tall, and
    # have an aspect ratio approximately equal to 1
    if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
        questionCnts.append(c)

# sort the question contours top-to-bottom, then initialize
# the total number of correct answers
questionCnts = cont.sort_contours(questionCnts, method="top-to-bottom")
correct = 0
 
# each question has 5 possible answers, to loop over the
# question in batches of 5
for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
    # sort the contours for the current question from
    # left to right, then initialize the index of the
    # bubbled answer
    cnts = cont.sort_contours(questionCnts[i:i + 5])[0]
    bubbled = None

        # loop over the sorted contours
for (j, c) in enumerate(cnts):
    # construct a mask that reveals only the current
    # "bubble" for the question
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv.drawContours(mask, [c], -1, 255, -1)

    # apply the mask to the thresholded image, then
    # count the number of non-zero pixels in the
    # bubble area
    mask = cv.bitwise_and(thresh, thresh, mask=mask)
    total = cv.countNonZero(mask)

# if the current total has a larger number of total
# non-zero pixels, then we are examining the currently
# bubbled-in answer
if bubbled is None or total > bubbled[0]:
    bubbled = (total, j)
# initialize the contour color and the index of the
# *correct* answer
color = (0, 0, 255)
k = ANSWER_KEY[q]

# check to see if the bubbled answer is correct
if k == bubbled[1]:
    color = (0, 255, 0)
    correct += 1

# draw the outline of the correct answer on the test
cv.drawContours(paper, [cnts[k]], -1, color, 3)

cv.imwrite("test1.png", paper)
cv.imwrite("test2.png", warped)
sys.exit()
