import cv2
import numpy as np
import csv

csvFileName = "datasets/poses.csv"
print 'Running ImageMosaic by Alex Hagiopol.'
imageData = []

print "Reading image data from %s." %csvFileName
with open(csvFileName) as csvFile:
    csvReader = csv.reader(csvFile, delimiter=',')
    for row in csvReader:
        imageData.append(row)
print "Read data for %i images." % len(imageData)



