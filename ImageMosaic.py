import utilities as util
import Combiner
import cv2
import os

fileName = "datasets/imageData.txt"
imageDirectory = "datasets/images/"
os.mkdir("results")
dataMatrix, allImages = util.importData(fileName, imageDirectory)
myCombiner = Combiner.Combiner(allImages,dataMatrix)
result = myCombiner.createMosaic()
util.display("RESULT",result)
cv2.imwrite("results/finalResult.png",result)



















