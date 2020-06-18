import numpy as np
import matplotlib.pyplot as plt
import pandas
import cv2
import glob
import argparse
import os
import shapely
import datetime
import shapely.geometry as geom
from shapely.ops import nearest_points
from tqdm import tqdm
import shutil

'''

   This command line tool will extract the minimal distance between the fish body and the interface.
   It will need the tracking result from FastTrack, and the minimal and maximal projection.

   It can take directly a list of path to the image sequences.

   The tool will create a folder named control alongside the image sequences.
  The fish and the minimal distance between the fish and the interface will be displayed.

'''




def extractInterface(image, minImage, maxImage, data, param, thresh):

      # Match tracking ROI, need FastTRack version > 4
      try:
        roi = [int(param[param[0] == "ROI top x"][1].values), int(param[param[0] == "ROI top y"][1].values), int(param[param[0] == "ROI bottom x"][1].values), int(param[param[0] == "ROI bottom y"][1].values)]
        if roi[2] == 0:
            roi[2] = minImage.shape[1]
        if roi[3] == 0:
            roi[3] = minImage.shape[0]
        maxArea = float(param[param[0] == "Maximal size"][1].values)

      except:
        roi = [5, 50, image.shape[1] - 10, image.shape[0] - 50]
        maxArea = 9000


      # Normalization by min and max projection
      sub = np.copy(image)
      sub = ( np.float32(image) - np.float32(minImage) ) / (np.float32(maxImage) - np.float32(minImage))
      sub = sub.clip(0, 1) 
      sub *= 255
      sub = np.uint8(sub)
      sub = sub[roi[1]:roi[3], roi[0]:roi[2]]
      base = np.copy(sub)
      image = image[roi[1]:roi[3], roi[0]:roi[2]]
      
      
      # Fish detection: Select the tracking data corresponding with the current image.
      # Select a sub image containing only the fish and extract the contour.
      # Append the contour and the associated id in the object stack.
      obj = []
      for row in data.itertuples():
        # Take a sub image containing each fish
        y = np.array((row.yBody - roi[1]  - 150, row.yBody - roi[1] + 150))
        yOffset = 0
        if y[0] < 0:
          yOffset -= int(y[0])
        y = y.clip(0, sub.shape[0])
        x = np.array((row.xBody - roi[0] - 150, row.xBody - roi[0] + 150))
        xOffset = 0
        if x[0] < 0:
          xOffset -= int(x[0])
        x = x.clip(0, sub.shape[1])
        fish = image[int(y[0]):int(y[1]), int(x[0]):int(x[1])] - maxImage[roi[1]:roi[3], roi[0]:roi[2]][int(y[0]):int(y[1]), int(x[0]):int(x[1])]
        __, bina = cv2.threshold(fish, 190, 255, cv2.THRESH_BINARY_INV)
        #_, bina = cv2.threshold(fish, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cnts, __ = cv2.findContours(bina, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fishCnt = [i for i in cnts if (cv2.contourArea(i)>15 and cv2.contourArea(i)<maxArea)]
        fishCnt = sorted(fishCnt, key=lambda x: cv2.contourArea(x))
        if len(fishCnt) > 0:
          # Check which contour is the fish
          for i, j in enumerate(fishCnt):
            if cv2.pointPolygonTest(j, (150 - xOffset, 150 - yOffset), measureDist=False) == 1:
              obj.append((row.id, j))
              break


      # Interface detection
      kernel = np.ones((9, 9), np.uint8) 
      dst = cv2.morphologyEx(sub, cv2.MORPH_DILATE, kernel, iterations=1) 
      if not thresh:
          __, dst = cv2.threshold(dst, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      else:
        __, bina = cv2.threshold(fish, int(thresh), 255, cv2.THRESH_BINARY_INV)
      dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel, iterations=4) 
      dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel, iterations=4) 
      cnts, __ = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)
      #base = np.copy(dst)
      
      # Interface cleaning, remove horizontal edges.
      for i, j in enumerate(cnts[0]):
          if j[0][0] < 200:
              if j[0][1] <= 200:
                  j[0][1] = -5000
                  j[0][0] = -5000
              else:
                  j[0][1] = 5000
                  j[0][0] = -5000
          elif j[0][0] > 800:
              if j[0][1] <= 200:
                  j[0][1] = -5000
                  j[0][0] = 5000
              else:
                  j[0][1] = 5000
                  j[0][0] = 5000

          if j[0][1] < 5: 
                  j[0][1] = -5000
          if j[0][1] > roi[3] - roi[1] - 5:
                  j[0][1] = 5000
      

      # Convert cv2 contours to shapely objects
      interfaceShape = []
      for i in cnts[0]:
          if abs(i[0][0]) < 2000 and abs(i[0][1]) < 2000:
              interfaceShape.append((i[0][0], i[0][1]))
      interfaceShape = np.asarray(interfaceShape)
      
      objectShapes = []
      for o in obj:
        fishShape = []
        tmpData = data[data.id == o[0]]
        x = np.array((int(tmpData.xBody) - roi[0] - 150, int(tmpData.xBody) - roi[0] + 150))
        y = np.array((int(tmpData.yBody) - roi[1] - 150, int(tmpData.yBody) - roi[1] + 150))
        y = y.clip(0, sub.shape[0])
        x = x.clip(0, sub.shape[1])
        for i in o[1]:
            fishShape.append((i[0][0] + x[0], i[0][1] + y[0]))
        fishShape = np.asarray(fishShape)
        objectShapes.append((o[0], fishShape))


      #Compute the minimal distance between the fish contour and the interface
      distances = []
      for i, j in objectShapes:
        if j.size > 2 and interfaceShape.size > 2:
            fishLine = geom.LineString(j)
            interfaceLine = geom.LineString(interfaceShape)
            dist =  fishLine.distance(interfaceLine)
            nep = nearest_points(fishLine, interfaceLine)
            if nep[0].x < nep[1].x:
                dist*=-1
            distances.append((i, dist))

      return base, distances, interfaceShape, objectShapes




def extractDist(path, thresh=None):

  images = glob.glob(path + "/Frame*pgm")
  images.sort()
  tracking = pandas.read_csv(path + "/Tracking_Result/tracking.txt", sep='\t')
  meta = pandas.read_csv(path + "/Milestones.txt", sep='\t', header=None)
  try:
    param = pandas.read_csv(path + "/Tracking_Result/parameter.param", sep = ' = ', header = None)
  except: 
    param = pandas.DataFrame(columns = ['id'])
    print("Error opening param")

  minImage = cv2.imread(path + "/minProjection.pgm", flags = cv2.IMREAD_GRAYSCALE)
  maxImage = cv2.imread(path + "/maxProjection.pgm", flags = cv2.IMREAD_GRAYSCALE)
  interfaces = []
  distances = []
  fish = []
  firstCycle = (meta[0][3], meta[0][4])
  secondCycle =(meta[0][7], len(images))

  with open(path + "/distance.txt", "w") as outFile:
    outFile.write("imageNumber" + '\t' + "distance"  + '\t' + "id" + '\n')
    try :
      shutil.rmtree(path + "/control/")
      os.mkdir(path + "/control/")
    except:
      os.mkdir(path + "/control/")

    plt.figure()
    for i, j in tqdm(enumerate(images), ascii=True, total=len(images)):
        if not tracking[tracking.imageNumber == i].empty:
            dat = tracking[tracking.imageNumber == i]
            if i in range(firstCycle[0], firstCycle[1]):
                img, distances, interfaces, objects = extractInterface(cv2.imread(j, flags=cv2.IMREAD_GRAYSCALE), minImage, maxImage, dat, param, thresh)
                for k, l in distances:
                  outFile.write(str(i) + '\t' + str(l) + '\t' + str(k) + '\n')

                plt.imshow(img)
                for d, di in zip(objects, distances):
                  if interfaces.size > 2 and d[1].size > 2:
                    plt.plot(*interfaces.T, "r")
                    plt.plot(*d[1].T)
                    line0 = geom.LineString(interfaces)
                    line1 = geom.LineString(d[1])
                    nep = nearest_points(line0, line1)
                    plt.plot([nep[0].x, nep[1].x], [nep[0].y, nep[1].y], color='red', marker='o', scalex=False, scaley=False)
                    #plt.xlim(0,1000)
                    #plt.ylim(500,0)
                plt.savefig(path + "/control/" + str(i) + ".png", dpi=100)
                plt.clf()
            elif i in range(secondCycle[0], secondCycle[1]):
                img, distances, interfaces, objects = extractInterface(cv2.imread(j, flags=cv2.IMREAD_GRAYSCALE), minImage, maxImage, dat, param, thresh)
                for k, l in distances:
                  outFile.write(str(i) + '\t' + str(l) + '\t' + str(k) + '\n')

                plt.imshow(img)
                for d, di in zip(objects, distances):
                  if interfaces.size > 2 and d[1].size > 2:
                    plt.plot(*interfaces.T, "r")
                    plt.plot(*d[1].T)
                    line0 = geom.LineString(interfaces)
                    line1 = geom.LineString(d[1])
                    nep = nearest_points(line0, line1)
                    plt.plot([nep[0].x, nep[1].x], [nep[0].y, nep[1].y], color='red', marker='o', scalex=False, scaley=False)
                    #plt.xlim(0,1000)
                    #plt.ylim(500,0)
                plt.savefig(path + "/control/" + str(i) + ".png", dpi=100)
                plt.clf()



parser = argparse.ArgumentParser(description="Extract the closest distance between interface and fish") 
parser.add_argument("path", nargs='+', help="Path to the folder of the experiment")
parser.add_argument("--thresh", dest="thresh", default=None, help="Set a manual threshold, default is automatic Otsu")
args = parser.parse_args()
success = []
failure = []
for i in args.path:
  try:
     extractDist(i, thresh=args.thresh)
     success.append(i)
  except Exception as e:
     failure.append(i)
     print(e)

name = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
with open(name + ".log", 'w') as f:
  f.write(name + " extractDist\n")
  for i in success:
    f.write(i + " Done\n")
  for i in failure:
    f.write(i + " Fail\n")
