import cv2
import numpy as np
import utilities as util
import geometry as gm
import pyramidBlending
import copy

class Combiner:
    def __init__(self,imageList_,dataMatrix_):
        self.imageList = []
        self.dataMatrix = dataMatrix_
        detector = cv2.ORB()
        for i in range(0,len(imageList_)):
            image = imageList_[i][::2,::2,:]
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            M = gm.computeUnRotMatrix(self.dataMatrix[i,:])
            correctedImage = gm.warpPerspectiveWithPadding(imageList_[i],M)
            self.imageList.append(correctedImage)
        self.resultImage = self.imageList[0]
    def createMosaic(self):
        for i in range(1,len(self.imageList)):
            self.combine(i)
        return self.resultImage

    def combine(self, index2):
        '''
        :param index2: index of self.imageList and self.kpList to combine with self.referenceImage and self.referenceKeypoints
        :return: combination of reference image and image at index 2
        '''

        image1 = copy.copy(self.imageList[index2 - 1])
        image2 = copy.copy(self.imageList[index2])
        #kp1 = copy.copy(self.referenceKP)
        #kp2 = copy.copy(self.kpList[index2])

        '''Descriptor computation and matching'''
        detector = cv2.SURF(2000) #cv2.ORB(1000)
        gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
        ret1, mask1 = cv2.threshold(gray1,1,255,cv2.THRESH_BINARY)
        kp1, descriptors1 = detector.detectAndCompute(gray1,mask1)

        gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
        ret2, mask2 = cv2.threshold(gray2,1,255,cv2.THRESH_BINARY)
        kp2, descriptors2 = detector.detectAndCompute(gray2,mask2)

        '''
        if index2 > 1:
            gray_p = cv2.cvtColor(self.imageList[index2-1],cv2.COLOR_BGR2GRAY)
            ret_p, mask_p = cv2.threshold(gray_p,1,255,cv2.THRESH_BINARY)
            kp_p, descriptors_p = detector.detectAndCompute(gray_p,mask_p)

            kp1 = kp1 + kp_p
            descriptors2 = descriptors2 + descriptors_p
        '''
        test = cv2.drawKeypoints(image1,kp1,color=(0,0,255))
        util.display("TEST",test)
        test = cv2.drawKeypoints(image2,kp2,color=(0,0,255))
        util.display("TEST",test)

        matcher = cv2.BFMatcher()#cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
        matches = matcher.knnMatch(descriptors2,descriptors1, k=2)#matcher.match(descriptors2,descriptors1)
        #prune bad matches
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)
        matches = copy.copy(good)

        matchDrawing = util.drawMatches(gray2,kp2,gray1,kp1,matches)
        util.display("matches",matchDrawing)

        src_pts = np.float32([ kp2[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp1[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

        '''Compute Affine Transform'''
        A = cv2.estimateRigidTransform(src_pts,dst_pts,fullAffine=False)
        if A == None:
            print "FINDING AFFINE TRANSFORM FAILED. ATTEMPT FULL HOMOGRAPHY"
            HomogResult = cv2.findHomography(src_pts,dst_pts,method=cv2.RANSAC)
            H = HomogResult[0]
        '''Compute 4 Image Corners Locations'''
        height1,width1 = image1.shape[:2]
        height2,width2 = image2.shape[:2]
        corners1 = np.float32(([0,0],[0,height1],[width1,height1],[width1,0]))
        corners2 = np.float32(([0,0],[0,height2],[width2,height2],[width2,0]))
        warpedCorners2 = np.zeros((4,2))
        for i in range(0,4):
            cornerX = corners2[i,0]
            cornerY = corners2[i,1]
            if A != None:
                warpedCorners2[i,0] = A[0,0]*cornerX + A[0,1]*cornerY + A[0,2]
                warpedCorners2[i,1] = A[1,0]*cornerX + A[1,1]*cornerY + A[1,2]
            else:
                warpedCorners2[i,0] = (H[0,0]*cornerX + H[0,1]*cornerY + H[0,2])/(H[2,0]*cornerX + H[2,1]*cornerY + H[2,2])
                warpedCorners2[i,1] = (H[1,0]*cornerX + H[1,1]*cornerY + H[1,2])/(H[2,0]*cornerX + H[2,1]*cornerY + H[2,2])
        allCorners = np.concatenate((corners1, warpedCorners2), axis=0)
        [xMin, yMin] = np.int32(allCorners.min(axis=0).ravel() - 0.5)
        [xMax, yMax] = np.int32(allCorners.max(axis=0).ravel() + 0.5)

        '''Compute Image Alignment and Keypoint Alignment'''
        translation = np.float32(([1,0,-1*xMin],[0,1,-1*yMin],[0,0,1]))
        warpedResImg = cv2.warpPerspective(self.resultImage, translation, (xMax-xMin, yMax-yMin))
        if A == None:
            fullTransformation = np.dot(translation,H)
            warpedImage2 = cv2.warpPerspective(image2, fullTransformation, (xMax-xMin, yMax-yMin))
        else:
            warpedImageTemp = cv2.warpPerspective(image2, translation, (xMax-xMin, yMax-yMin))
            warpedImage2 = cv2.warpAffine(warpedImageTemp, A, (xMax-xMin, yMax-yMin))
        self.imageList[index2] = copy.copy(warpedImage2) #crucial: update old images for future feature extractions

        '''
        #crucial: update old images for future feature extractions
        for i in range(0,index2 + 1)
            self.imageList[i] =
        '''
        resGray = cv2.cvtColor(self.resultImage,cv2.COLOR_BGR2GRAY)
        warpedResGray = cv2.warpPerspective(resGray, translation, (xMax-xMin, yMax-yMin))

        '''Compute Mask for Image Combination'''
        ret, mask1 = cv2.threshold(warpedResGray,1,255,cv2.THRESH_BINARY_INV)
        mask3 = np.float32(mask1)/255

        warpedImage2[:,:,0] = warpedImage2[:,:,0]*mask3
        warpedImage2[:,:,1] = warpedImage2[:,:,1]*mask3
        warpedImage2[:,:,2] = warpedImage2[:,:,2]*mask3

        result = warpedResImg + warpedImage2
        #self.referenceImage = result
        self.resultImage = result
        util.display("result",result)
        return result

