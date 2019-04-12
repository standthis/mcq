


#---- 4 corner points of the bounding box
#pts_src = np.array([[17.0,0.0], [77.0,5.0], [0.0, 552.0],[53.0, 552.0]])
pts_src = docCnt

#---- 4 corner points of the black image you want to impose it on
#pts_dst = np.array([[0.0,0.0],[77.0, 0.0],[ 0.0,552.0],[77.0, 552.0]])
pts_dst = docCnt

#---- forming the black image of specific size
#im_dst = np.zeros((552, 77, 3), np.uint8)
im_dst = np.zeros_like(warped)

#---- Framing the homography matrix
h, status = cv.findHomography(pts_src, pts_dst)

#---- transforming the image bound in the rectangle to straighten
im_out = cv.warpPerspective(warped, h, (im_dst.shape[1],im_dst.shape[0]))
cv.imwrite("warped.png", warped)
cv.imwrite("im_out.png", im_out)
sys.exit()
#cv.imwrite("im_out.jpg", im_out)