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
result = myCombiner.createMosaic()
util.display("RESULT",result)



















