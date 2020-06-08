import numpy as np
import pandas
import cv2
import glob
import argparse
import os
import datetime
from tqdm import tqdm


'''
   This command line tool will compute from an image sequence the maximum and minimum projection.
   The image sequences must be given with the tracking result from FastTrack.
   The fish will be removed from the image to achievied the minimum projection.
   Small artefact can be also removed

   A list of path can directly be given with the number of images to take as one of n image taken.

'''


def minMax(path, n, paint = True):
    imageList = glob.glob(path + "/Frame*pgm")
    tracking = pandas.read_csv(path + "/Tracking_Result/tracking.txt", sep='\t')
    imageList.sort()
    image = cv2.imread(imageList[0], flags=cv2.IMREAD_GRAYSCALE)
    maxImage = np.zeros(image.shape, dtype = "uint8")
    minImage = np.ones(image.shape, dtype = "uint8")*255
    
    for i in tqdm(range(0, len(imageList), n)):
        tmp = cv2.imread(imageList[i], flags=cv2.IMREAD_GRAYSCALE)
        maxImage = cv2.max(tmp, maxImage)


        if not tracking[tracking.imageNumber == i].empty:
          for row in tracking[tracking.imageNumber == i].itertuples():
            dat = (int(row.xBody), int(row.yBody))
            size = 200
            y = np.array((dat[1] - size, dat[1] + size))
            y = y.clip(0, image.shape[0])
            x = np.array((dat[0] - size, dat[0] + size))
            x = x.clip(0, image.shape[1])
            sub = tmp[y[0]:y[1], x[0]:x[1]]
            tmp[y[0]:y[1], x[0]:x[1]] = np.ones(sub.shape)*255
          minImage = cv2.min(tmp, minImage)

    param = pandas.read_csv(path + "/Tracking_Result/parameter.param", sep = ' = ', header = None)
    roi = [int(param[param[0] == "ROI top x"][1].values), int(param[param[0] == "ROI top y"][1].values), int(param[param[0] == "ROI bottom x"][1].values), int(param[param[0] == "ROI bottom y"][1].values)]
    if roi[2] == 0:
        roi[2] = minImage.shape[1]
    if roi[3] == 0:
        roi[3] = minImage.shape[0]

    if paint:
        kernel = np.ones((11, 11), np.uint8) 
        tmp = minImage-maxImage
        tmp = tmp[roi[1]:roi[3], roi[0]:roi[2]]
        __, dst = cv2.threshold(tmp, 200, 255, cv2.THRESH_BINARY_INV)
        dst = cv2.morphologyEx(dst, cv2.MORPH_DILATE, kernel, iterations=3) 
        minImage[roi[1]:roi[3], roi[0]:roi[2]] = cv2.inpaint(minImage[roi[1]:roi[3], roi[0]:roi[2]], dst, 100, cv2.INPAINT_NS)


    cv2.imwrite(path + "/maxProjection.pgm", maxImage)
    cv2.imwrite(path + "/minProjection.pgm", minImage)
        

parser = argparse.ArgumentParser(description="Extract the maximal and minimal z-projection") 
parser.add_argument("path", nargs='+', help="Path to the folder of the experiment")
parser.add_argument("-n", dest="number", help="One on n images taken")
parser.add_argument("--paint", dest="paint", action='store_true', help="Inpainting the min image to correct artifacts")
args = parser.parse_args()
success = []
failure = []

for i in args.path:
  try:
     minMax(i, int(args.number), args.paint)
     success.append(i)
  except Exception as e:
     failure.append(i)
     print(i, e)

name = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
with open(name + ".log", 'w') as f:
  for i in success:
    f.write(i + " Done\n")
  for i in failure:
    f.write(i + " Fail\n")


