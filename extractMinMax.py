import numpy as np
import pandas
import cv2
import glob
import argparse
import os
from tqdm import tqdm


'''
   This command line tool will compute from an image sequence the maximum and minimum projection.
   The image sequences must be given with the tracking result from FastTrack.
   The fish will be removed from the image to achievied the minimum projection.
   Small artefact can be also removed

   A list of path can directly be given with the number of images to take as one of n image taken.

'''


def minMax(path, n):
    imageList = glob.glob(path + "/Frame*pgm")
    tracking = pandas.read_csv(path + "/Tracking_Result/tracking.txt", sep='\t')
    imageList.sort()
    image = cv2.imread(imageList[0], flags=cv2.IMREAD_GRAYSCALE)
    maxImage = np.zeros(image.shape, dtype = "uint8")
    minImage = np.ones(image.shape, dtype = "uint8")*255
    
    for i in tqdm(range(0, len(imageList), n)):
        tmp = cv2.imread(imageList[i], flags=cv2.IMREAD_GRAYSCALE)
        maxImage = cv2.max(tmp, maxImage)

        if not tracking["xBody"][tracking.imageNumber == i].empty:
          dat = (int(tracking["xBody"][tracking.imageNumber == i]), int(tracking["yBody"][tracking.imageNumber == i]))
          y = np.array((dat[1] - 200, dat[1] + 200))
          y = y.clip(0, image.shape[0])
          x = np.array((dat[0] - 200, dat[0] + 200))
          x = x.clip(0, image.shape[1])
          sub = tmp[y[0]:y[1], x[0]:x[1]]
          tmp[y[0]:y[1], x[0]:x[1]] = np.ones(sub.shape)*255
          minImage = cv2.min(tmp, minImage)

        
    minCrop = minImage[50:450, 10:990]
    __, bina = cv2.threshold(minCrop, 90, 1, cv2.THRESH_BINARY_INV)
    kernel = np.ones((9,9), np.uint8) 
    bina = cv2.dilate(bina, kernel, iterations=1) 
    minCrop = cv2.inpaint(minCrop, bina, 3, cv2.INPAINT_NS)
    minImage[50:450, 10:990] = minCrop

    cv2.imwrite(path + "/maxProjection.pgm", maxImage)
    cv2.imwrite(path + "/minProjection.pgm", minImage)
        

parser = argparse.ArgumentParser(description="Extract the maximal and minimal z-projection") 
parser.add_argument("path", nargs='+', help="Path to the folder of the experiment")
parser.add_argument("-n", dest="number", help="One on n images taken")
args = parser.parse_args()
for i in args.path:
  try:
    minMax(i, int(args.number))
    print("Done " + i)
  except Exception as e:
    print("Error for " + str(i), e)
    pass

