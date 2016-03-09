import utilities as util
import Combiner
import geometry as gm
import cv2
import numpy as np
import copy

fileName = "datasets/imageData.txt"
imageDirectory = "datasets/images/"
dataMatrix, allImages = util.importData(fileName, imageDirectory)
myCombiner = Combiner.Combiner(allImages,dataMatrix)











'''
myCombiner = Combiner.Combiner(allImages,dataMatrix)
combinedResult = myCombiner.createMosaic()
util.display("COMBINED RESULT", combinedResult)
'''










'''
stitched = paddedImg1/2 + warpedImg2/2
cv2.namedWindow("stitched",cv2.WINDOW_NORMAL)
cv2.resizeWindow("stitched",1920,1080)
cv2.imshow("stitched",stitched)
cv2.waitKey(0)
'''
'''
Feature detection and matching
'''
'''
detector = cv2.ORB()
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#for i in range(0,len(allImages)-1):
image1 = allImages[0]
gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
keypoints1, descriptors1 = detector.detectAndCompute(gray1,None)
#result1 = cv2.drawKeypoints(image1,keypoints1,color=(0,0,255))

image2 = allImages[1]
gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
keypoints2, descriptors2 = detector.detectAndCompute(gray2,None)
#result2 = cv2.drawKeypoints(image2,keypoints2,color=(0,0,255))

matches = matcher.match(descriptors2,descriptors1)
'''
'''
Compute Homography
'''
'''
#Homography to warp image2 into image 1
src_pts = np.float32([ keypoints2[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
dst_pts = np.float32([ keypoints1[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
H = cv2.findHomography(src_pts,dst_pts,method=cv2.RANSAC)
Homog21 = H[0] #3x3 homography matrix from img2 to img1

warped = cv2.warpPerspective(image2,Homog21,(image1.shape[1],image2.shape[0]),borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
stitched = image1/2 + warped/2
cv2.namedWindow("stitched",cv2.WINDOW_NORMAL)
cv2.resizeWindow("stitched",1920,1080)
cv2.imshow("stitched",stitched)
cv2.waitKey(0)
'''


















