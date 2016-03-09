import cv2
import numpy as np


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
    cv2.destroyWindow(title)

def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for m in matches:

        # Get the matching keypoints for each of the images
        img1_idx = m.queryIdx
        img2_idx = m.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        radius = 8
        thickness = 3
        color = (255,0,0) #blue
        cv2.circle(out, (int(x1),int(y1)), radius, color, thickness)
        cv2.circle(out, (int(x2)+cols1,int(y2)), radius, color, thickness)

        # Draw a line in between the two points
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), color, thickness)

    # Also return the image if you'd like a copy
    return out



    #fullTransformation = np.dot(A,translation)

    #H = cv2.findHomography(src_pts,dst_pts,method=cv2.RANSAC,ransacReprojThreshold=0.1)
    #Homog21 = H[0] #3x3 homography matrix from img2 to img1

    #Homog21 = np.float32(([A[0,0],A[0,1],A[0,2]],[A[1,0],A[1,1],A[1,2]],[0,0,1]))

    #corners1 = np.float32([[0,0],[0,height1],[width1,height1],[width1,0]]).reshape(-1,1,2)
    #corners2 = np.float32([[0,0],[0,height2],[width2,height2],[width2,0]]).reshape(-1,1,2)
    #warpedCorners2 = cv2.warpAffine(corners2,A,(corners2.shape[1],corners2.shape[0]),)#cv2.perspectiveTransform(corners2, Homog21)


    '''
    gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(gray1,None)
    kp2, des2 = sift.detectAndCompute(gray2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des2,des1, k=2)
    '''
    '''
    # Apply ratio test
    goodMatches = []
    for m,n in matches:
        #print m
        #print n
        if m.distance < 0.75*n.distance:
            goodMatches.append(m)
    '''

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
