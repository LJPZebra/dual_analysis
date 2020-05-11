import numpy as np
import matplotlib.pyplot as plt
import pandas
import cv2
import glob
import argparse
import os
import shapely
import shapely.geometry as geom
from shapely.ops import nearest_points
from tqdm import tqdm
import shutil

'''

   This command line tool will extract the minimal distance between the fish body and the interface.
   It will need the tracking result from FastTrack, and the minimal and maximal projection.

   It can take directly a list of path to the image sequences.

   The tool will create a folder control alongside the images where the two sides, the fish and the minimal
   distance between the fish and the interface will be displayed.

'''




def extractInterface(image, minImage, maxImage, distances, interfaces, objects, dat, datHead, param):
    try:
      roi = [10, 50, image.shape[1] - 10, image.shape[0] - 50]

#Normalization
      sub = np.copy(image)
      paintCrop = np.copy(sub[roi[1]:roi[3], roi[0]:roi[2]])
      __, binaPaint = cv2.threshold(paintCrop, 90, 1, cv2.THRESH_BINARY_INV)
      kernel = np.ones((11, 11), np.uint8) 
      binaPaint = cv2.dilate(binaPaint, kernel, iterations=2) 
      paintCrop = cv2.inpaint(paintCrop, binaPaint, 15, cv2.INPAINT_NS)
      sub[roi[1]:roi[3], roi[0]:roi[2]] = paintCrop

      sub = ( np.float32(image) - np.float32(minImage) ) / (np.float32(maxImage) - np.float32(minImage))
      sub = sub.clip(0, 1) 
      sub *= 255
      sub = np.uint8(sub)
      sub = sub[roi[1]:roi[3], roi[0]:roi[2]]
      image = image[roi[1]:roi[3], roi[0]:roi[2]]
      
      
      # Fish inpainting
      y = np.array((dat[1] - roi[1]  - 100, dat[1] - roi[1] + 100))
      y = y.clip(0, sub.shape[0])
      x = np.array((dat[0] - 100 - roi[0], dat[0] - roi[0] + 100))
      x = x.clip(0, sub.shape[1])
      fish = np.copy(image[y[0]:y[1], x[0]:x[1]])
      #fishNorma = np.copy(sub[y[0]:y[1], x[0]:x[1]])
      #__, bina = cv2.threshold(fish, 90, 1, cv2.THRESH_BINARY_INV)
      #kernel = np.ones((9,9), np.uint8) 
      #bina = cv2.dilate(bina, kernel, iterations=1) 
      #fishInpaint = cv2.inpaint(fishNorma, bina, 25, cv2.INPAINT_NS)
      #sub[y[0]:y[1], x[0]:x[1]] = fishInpaint

      '''paintCrop = sub[50:450, 10:990]
      __, binaPaint = cv2.threshold(paintCrop, 20, 1, cv2.THRESH_BINARY_INV)
      kernel = np.ones((9,9), np.uint8) 
      binaPaint = cv2.dilate(binaPaint, kernel, iterations=1) 
      paintCrop = cv2.inpaint(paintCrop, binaPaint, 7, cv2.INPAINT_NS)
      sub[50:450, 10:990] = paintCrop'''
      
      # Fish detection
      __, bina = cv2.threshold(fish, 90, 1, cv2.THRESH_BINARY_INV)
      cnts, __ = cv2.findContours(bina, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      fishCnt = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)[0]

      # Interface detection
      kernel = np.ones((9, 9), np.uint8) 
      dst = cv2.morphologyEx(sub, cv2.MORPH_DILATE, kernel, iterations=2) 
      __, dst = cv2.threshold(dst, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel, iterations=4) 
      dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel, iterations=4) 
      cnts, __ = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)
      
      # Interface cleaning
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
          if j[0][1] > sub.shape[0] - 5:
                  j[0][1] = 5000
      
      interfaceShape = []
      for i in cnts[0]:
          if abs(i[0][0]) < 2000 and abs(i[0][1]) < 2000:
              interfaceShape.append((i[0][0], i[0][1]))
      interfaceShape = np.asarray(interfaceShape)

      
      fishShape = []
      for i in fishCnt:
              fishShape.append((i[0][0] + x[0], i[0][1] + y[0]))
      fishShape = np.asarray(fishShape)

#Minimal distance
      if fishShape.size != 0 and interfaceShape.size != 0:
          fishLine = geom.LineString(fishShape)
          interfaceLine = geom.LineString(interfaceShape)
          dist =  fishLine.distance(interfaceLine)
          nep = nearest_points(fishLine, interfaceLine)
          if nep[0].x < nep[1].x:
              dist*=-1
          distances.append(dist)
          interfaces.append(interfaceShape)
          objects.append(fishShape)
      else:
          distances.append(np.nan)
          interfaces.append(np.array((np.nan)))
          objects.append(np.array((np.nan)))
      return dst
    except Exception as e:
      print(e)
      distances.append(np.nan)
      interfaces.append(np.array((np.nan)))
      objects.append(np.array((np.nan)))

      return image




def extractDist(path):

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
  firstCycle = (meta[0][3] + 2000, meta[0][4])
  secondCycle =(meta[0][7] + 2000, len(images))

  with open(path + "/distance.txt", "w") as outFile:
    outFile.write("imageNumber" + '\t' + "distance" + '\n')
    try :
      shutil.rmtree(path + "/control/")
      os.mkdir(path + "/control/")
    except:
      os.mkdir(path + "/control/")

    plt.figure()
    for i, j in tqdm(enumerate(images), ascii=True, total=len(images)):
        if not tracking["xBody"][tracking.imageNumber == i].empty:
            dat = (int(tracking["xBody"][tracking.imageNumber == i]), int(tracking["yBody"][tracking.imageNumber == i]))
            datHead = (int(tracking["xHead"][tracking.imageNumber == i]), int(tracking["yHead"][tracking.imageNumber == i]), int(tracking["tHead"][tracking.imageNumber == i]))
            if i in range(firstCycle[0], firstCycle[1]):
                img = extractInterface(cv2.imread(j, flags=cv2.IMREAD_GRAYSCALE), minImage, maxImage, distances, interfaces, fish, dat, datHead, param)
                if not np.isnan(distances[-1]):
                  plt.imshow(img)
                  plt.plot(*interfaces[-1].T, "r")
                  plt.plot(*fish[-1].T)
                  line0 = geom.LineString(interfaces[-1])
                  line1 = geom.LineString(fish[-1])
                  nep = nearest_points(line0, line1)
                  plt.plot([nep[0].x, nep[1].x], [nep[0].y, nep[1].y], color='red', marker='o', scalex=False, scaley=False)
                  plt.xlim(0,1000)
                  plt.ylim(400,0)
                  plt.title(str(distances[-1]))
                  plt.savefig(path + "/control/" + str(i) + ".png", dpi=100)
                  plt.clf()
            elif i in range(secondCycle[0], secondCycle[1]):
                img = extractInterface(cv2.imread(j, flags=cv2.IMREAD_GRAYSCALE), minImage, maxImage, distances, interfaces, fish, dat, datHead, param)
                if not np.isnan(distances[-1]):
                  plt.imshow(img)
                  plt.plot(*interfaces[-1].T)
                  plt.plot(*fish[-1].T)
                  line0 = geom.LineString(interfaces[-1])
                  line1 = geom.LineString(fish[-1])
                  nep = nearest_points(line0, line1)
                  plt.plot([nep[0].x, nep[1].x], [nep[0].y, nep[1].y], color='red', marker='o', scalex=False, scaley=False)
                  plt.xlim(0,1000)
                  plt.ylim(400,0)
                  plt.title(str(distances[-1]))
                  plt.savefig(path + "/control/" + str(i) + ".png", dpi=100)
                  plt.clf()
            else:
                distances.append(np.nan)
                interfaces.append(np.array((np.nan)))
                fish.append(np.array((np.nan)))
        else:
            distances.append(np.nan)
            interfaces.append(np.array((np.nan)))
            fish.append(np.array((np.nan)))

        outFile.write(str(i) + '\t' + str(distances[-1]) + '\n')


success = []
parser = argparse.ArgumentParser(description="Extract the closest distance between interface and fish") 
parser.add_argument("path", nargs='+', help="Path to the folder of the experiment")
args = parser.parse_args()
for i in args.path:
  try:
    extractDist(i)
    print("Done " + i)
    success.append(i)
  except Exception as e:
    print("Error for " + str(i), e)
    
print(success)
