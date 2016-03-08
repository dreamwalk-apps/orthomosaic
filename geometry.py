import numpy as np
import cv2
import math as m
import utilities as util

def computeUnRotMatrix(pose):
    '''
    See http://planning.cs.uiuc.edu/node102.html. Undoes the rotation of the craft relative to the world frame.
    Arguments:
        pose: A 1x6 NumPy ndArray containing pose information in [X,Y,Z,Y,P,R] format
    Returns:
        rot: A 3x3 rotation matrix that removes perspective distortion from the image to which it is applied.
    '''
    a = pose[3]*np.pi/180 #alpha
    b = pose[4]*np.pi/180 #beta
    g = pose[5]*np.pi/180 #gamma
    Rz = np.array(([m.cos(a), -1*m.sin(a),    0],
                   [m.sin(a),    m.cos(a),    0],
                   [       0,           0,     1]))

    Ry = np.array(([ m.cos(b),           0,     m.sin(b)],
                   [        0,           1,            0],
                   [-1*m.sin(b),           0,     m.cos(b)]))

    Rx = np.array(([        1,           0,            0],
                   [        0,    m.cos(g),  -1*m.sin(g)],
                   [        0,    m.sin(g),     m.cos(g)]))
    Ryx = np.dot(Rx,Ry)
    R = np.dot(Rz,Ryx)
    R[0,2] = 0
    R[1,2] = 0
    R[2,2] = 1
    Rtrans = R.transpose()
    InvR = np.linalg.inv(Rtrans)
    return InvR

def warpWithPadding(image,transformation):
    '''
    Produce a "padded" warped image that has black space on all sides so that warped image fits
    Arguments:
        image: ndArray image
        transformation: 3x3 ndArray representing perspective trransformation
    Returns:
        padded: ndArray image enlarged to exactly fit image warped by transformation
    '''

    height = image.shape[0]
    width = image.shape[1]
    corners = np.float32([[0,0],[0,height],[width,height],[width,0]]).reshape(-1,1,2)

    warpedCorners = cv2.perspectiveTransform(corners, transformation)
    [xMin, yMin] = np.int32(warpedCorners.min(axis=0).ravel() - 0.5)
    [xMax, yMax] = np.int32(warpedCorners.max(axis=0).ravel() + 0.5)
    translation = np.array(([1,0,-1*xMin],[0,1,-1*yMin],[0,0,1]))
    fullTransformation = np.dot(translation,transformation)
    result = cv2.warpPerspective(image, fullTransformation, (xMax-xMin, yMax-yMin))

    return result

def merge(refImage, image2,refImgContents):
    '''
    Adds second image to the first image. Assumes images have already been corrected.
    Arguments:
        image1 & image2: ndArrays already processed with computeUnRotMatrix() and warpWithPadding
        image1Contents: list of warped images that make up image 1
    Returns:
        result: second image warped into first image and combined with it
        warpedImage2: second image warped into first image but not combined
    '''

    '''Feature detection and matching'''
    image1 = refImgContents[len(refImgContents)-1]
    detector = cv2.ORB()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
    gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    kp1, descriptors1 = detector.detectAndCompute(gray1,None)
    gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
    kp2, descriptors2 = detector.detectAndCompute(gray2,None)
    matches = matcher.match(descriptors2,descriptors1)
    matchDrawing = util.drawMatches(gray1,kp1,gray2,kp2,matches)
    util.display("matches",matchDrawing)
    cv2.imwrite("matchDrawing.png",matchDrawing)
    src_pts = np.float32([ kp2[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp1[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    '''
    for i in range(1,len(refImgContents)):
        image1 = refImgContents[i]
        gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
        kp1, descriptors1 = detector.detectAndCompute(gray1,None)
        gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
        kp2, descriptors2 = detector.detectAndCompute(gray2,None)
        matches = matcher.match(descriptors2,descriptors1)
        util.matchDrawing = util.drawMatches(gray1,kp1,gray2,kp2,matches)
        util.display("matches",matchDrawing)
        src_pts = np.concatenate((src_pts,np.float32([ kp2[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)),axis=0)
        dst_pts = np.concatenate((dst_pts,np.float32([ kp1[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)),axis=0)
    '''
    #print src_pts
    '''Compute Affine Transform'''
    A = cv2.estimateRigidTransform(src_pts,dst_pts,fullAffine=False)
    print "A"
    print A

    '''Compute Corners'''
    height1,width1 = refImage.shape[:2]
    height2,width2 = image2.shape[:2]
    corners1 = np.float32(([0,0],[0,height1],[width1,height1],[width1,0]))
    corners2 = np.float32(([0,0],[0,height2],[width2,height2],[width2,0]))
    warpedCorners2 = np.zeros((4,2))
    for i in range(0,4):
        cornerX = corners2[i,0]
        cornerY = corners2[i,1]
        warpedCorners2[i,0] = A[0,0]*cornerX + A[0,1]*cornerY + A[0,2]
        warpedCorners2[i,1] = A[1,0]*cornerX + A[1,1]*cornerY + A[1,2]
    allCorners = np.concatenate((corners1, warpedCorners2), axis=0)
    [xMin, yMin] = np.int32(allCorners.min(axis=0).ravel() - 0.5)
    [xMax, yMax] = np.int32(allCorners.max(axis=0).ravel() + 0.5)

    '''Compute Image Alignment'''
    translation = np.float32(([1,0,-1*xMin],[0,1,-1*yMin],[0,0,1]))
    warpedRefImg = cv2.warpPerspective(refImage, translation, (xMax-xMin, yMax-yMin))
    warpedImageTemp = cv2.warpPerspective(image2, translation, (xMax-xMin, yMax-yMin))
    warpedImage2 = cv2.warpAffine(warpedImageTemp, A, (xMax-xMin, yMax-yMin))
    returnWarpedImage2 = np.copy(warpedImage2)

    refGray = cv2.cvtColor(refImage,cv2.COLOR_BGR2GRAY)
    warpedRefGray = cv2.warpPerspective(refGray, translation, (xMax-xMin, yMax-yMin))
    warpedGrayTemp = cv2.warpPerspective(gray2, translation, (xMax-xMin, yMax-yMin))
    warpedGray2 = cv2.warpAffine(warpedGrayTemp, A, (xMax-xMin, yMax-yMin))

    ret, mask1 = cv2.threshold(warpedRefGray,1,255,cv2.THRESH_BINARY_INV)
    ret, mask2 = cv2.threshold(warpedGray2,1,255,cv2.THRESH_BINARY_INV)

    mask1 = (np.float32(mask1)/255 + 1)/2
    mask2 = (np.float32(mask2)/255 + 1)/2

    warpedRefImg[:,:,0] = warpedRefImg[:,:,0]*mask2
    warpedRefImg[:,:,1] = warpedRefImg[:,:,1]*mask2
    warpedRefImg[:,:,2] = warpedRefImg[:,:,2]*mask2
    warpedImage2[:,:,0] = warpedImage2[:,:,0]*mask1
    warpedImage2[:,:,1] = warpedImage2[:,:,1]*mask1
    warpedImage2[:,:,2] = warpedImage2[:,:,2]*mask1
    result = warpedRefImg + warpedImage2
    util.display("result",result)
    cv2.imwrite("result.png",result)
    return result, returnWarpedImage2