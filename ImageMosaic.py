import utilities as util
import Combiner
import cv2

fileName = "datasets/imageData.txt"
imageDirectory = "datasets/images/"
dataMatrix, allImages = util.importData(fileName, imageDirectory)
myCombiner = Combiner.Combiner(allImages,dataMatrix)
result = myCombiner.createMosaic()
util.display("RESULT",result)
cv2.imwrite("results/finalResult.png",result)



















