def transform(src):
    expected = 847871.5
    image = src.copy()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edged = cv.Canny(blurred, 75, 200)
    cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE)
    cnts = conv.grab_contours(cnts)
    docCnt = None

    kernel = np.ones((5,5),np.uint8)
    closing = cv.morphologyEx(edged, cv.MORPH_CLOSE, kernel)
    dilation = cv.dilate(closing,kernel,iterations = 1)
    erosion = cv.erode(closing,kernel,iterations = 1)
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv.contourArea, reverse=True)
        for c in cnts:
            peri = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, 0.02 * peri, True)  
            center, radius = cv.minEnclosingCircle(approx)
            if len(approx) == 4:
                docCnt = approx
                break
    paper = four_point_transform(image, docCnt.reshape(4, 2))
    warped = four_point_transform(gray, docCnt.reshape(4, 2))
    return paper, warped
