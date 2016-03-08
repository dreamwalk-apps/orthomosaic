import cv2
import numpy as np
import math as m

def importData(fileName, imageDirectory):
    '''
    Arguments:
        fileName: Name of the pose data file in string form e.g. "datasets/imageData.txt"
        imageDirectory: Name of the directory where images arer stored in string form e.g. "datasets/images/"
    Returns:
        dataMatrix: A NumPy ndArray contaning all of the pose data. Each row stores 6 floats containing pose information in XYZYPR form
        allImages: A Python List of NumPy ndArrays containing images.
    '''

    allImages = [] #list of cv::Mat aimghes
    dataMatrix = np.genfromtxt(fileName,delimiter=",",usecols=range(1,7),dtype=float) #read numerical data
    fileNameMatrix = np.genfromtxt(fileName,delimiter=",",usecols=[0],dtype=str) #read filen name strings
    for i in range(0,fileNameMatrix.shape[0]): #read images
        allImages.append(cv2.imread(imageDirectory+fileNameMatrix[i]))
    print "Read data for %i images." % fileNameMatrix.shape[0]
    return dataMatrix, allImages

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

def display(title, image):
    '''
    OpenCV machinery for showing an image until the user presses a key.
    Arguments:
        title: Window title in string form
        image: ndArray containing image to show
    No returns.
    '''

    cv2.namedWindow(title,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title,1920,1080)
    cv2.imshow(title,image)
    cv2.waitKey(0)

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



    '''
    print "xMin %f" %xMin
    print "xMax %f" %xMax
    print "yMin %f" %yMin
    print "yMax %f" %yMax
    print "corners"
    print corners
    print "warped corners"
    print warpedCorners
    newWarpedCorners = cv2.perspectiveTransform(corners, fullTransformation)
    print "new warped corners"
    print newWarpedCorners
    '''

    '''
    Compute Corners
    '''
    '''
    #compute corner locations of undistorted rectangular image
    cornersImg = []
    cornersImg.append(np.array(([0],[0],[1])))
    cornersImg.append(np.array(([image.shape[0]],[0],[1])))
    cornersImg.append(np.array(([image.shape[0]],[image.shape[1]],[1])))
    cornersImg.append(np.array(([0],[image.shape[1]],[1])))
    '''
    '''
    #keep track of min and max row and column locations in warped image to know how much to pad original image
    minRLoc = 0
    maxRLoc = image.shape[0]
    minCLoc = 0
    maxCLoc = image.shape[1]
    #store warped corners
    warpedCornersImg = []
    for corner in cornersImg:
        warpedCorner = np.dot(transformation,corner)
        if warpedCorner[0] > maxRLoc:
            maxRLoc = warpedCorner[0]
        if warpedCorner[0] < minRLoc:
            minRLoc = warpedCorner[0]
        if warpedCorner[1] > maxCLoc:
            maxCLoc = warpedCorner[1]
        if warpedCorner[1] < minCLoc:
            minCLoc = warpedCorner[1]
        warpedCornersImg.append(warpedCorner)

    #pts1 = np.float32([[0,0],[0,image.shape[1]],[image.shape[0],image.shape[1]],[,0]]).reshape(-1,1,2)
    print "original corners"
    print cornersImg
    print "warped corners"
    print warpedCornersImg
    '''


    '''
    Compute Padding
    '''
    '''
    topPadding = 0
    bottomPadding = 0
    leftPadding = 0
    rightPadding = 0
    if minRLoc < 0:
        topPadding = abs(minRLoc)
    if minCLoc < 0:
        leftPadding = abs(minCLoc)
    if maxRLoc > image.shape[0]:
        bottomPadding = maxRLoc - image.shape[0]
    if maxCLoc > image.shape[1]:
        rightPadding = maxCLoc - image.shape[1]
    bottomPadding = bottomPadding + topPadding
    rightPadding = rightPadding + leftPadding
    print "minRLoc %f" % minRLoc
    print "top %f" % topPadding
    print "minCLoc %f" % minCLoc
    print "left %f" %leftPadding
    print "maxRLoc %f" % maxRLoc
    print "bottom %f" %bottomPadding
    print "maxCLoc %f" % maxCLoc
    print "right %f" %rightPadding

    xOffset = leftPadding
    yOffset = 0#topPadding
    translation = np.array(([1,0,xOffset],[0,1,yOffset],[0,0,1]))
    TranslatedHomog = np.dot(translation,transformation)

    paddedImg = cv2.copyMakeBorder(image,topPadding,bottomPadding,leftPadding,rightPadding,borderType=cv2.BORDER_CONSTANT,value=(0,0,0))
    warpedImg = cv2.warpPerspective(image,TranslatedHomog,(paddedImg.shape[1],paddedImg.shape[0]),borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
    return warpedImg
    '''

