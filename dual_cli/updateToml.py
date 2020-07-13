import pandas
import toml
import numpy as np
import argparse
import os
import datetime
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

def cleanDist(path, tracking, cycle1=None, cycle2=None):
    """
    This function will clean some artefact that are sometime present in the data.
    It will take the absolute difference in between the normalized distance signal and the normalized head position.
    Then all the values superior to 1.5*std are removed then interpoled.
    """
    with open(path + "/distanceCor.txt", 'w') as f:
        f.write("imageNumber\tdistance\tid\n")
    
    if not cycle1 or not cycle2:
        milestones = pandas.read_csv(path + "/Milestones.txt", sep='\t', header=None)
        cycle1 = (milestones[0][3], milestones[0][4])
        cycle2 = (milestones[0][6], np.max(tracking.imageNumber.values))
    
    data = pandas.read_csv(path + "/distance.txt", sep='\t')
    for fishInd, fishId in enumerate(set(tracking.id.values)):
        tmpData = data[data.id == fishId]
        tmpTracking = tracking[tracking.id == fishId]
        distance = []
        for i in tmpTracking.imageNumber.values:
            tmp = tmpData[tmpData.imageNumber == i]
            if not tmp.empty:
                distance.append(tmp.distance.values[0])
            else:
                distance.append(np.nan)
        distance = np.asarray(distance)
        
        xHead = tmpTracking.xHead.values - np.mean(tmpTracking.xHead.values)
        dist = distance - np.nanmean(distance)
        diff = np.abs((xHead - dist)**2)
        peaks = np.where(diff > (np.nanmean(diff) + 1*np.nanstd(diff)))[0]

        print("Values being removed " + str((len(peaks)/len(distance[~np.isnan(distance)]))*100) + " %")
        
        np.put(distance, peaks, np.nan)
        xInter = np.where(~np.isnan(distance))[0]
        if len(xInter) < 1:
          print(path)
          return False
        x = tmpTracking.imageNumber.values[xInter]
        y = distance[xInter]
        try:
          inter = interp1d(x, y, kind='cubic')
        except:
          print(path)
          return False
        
        minX = np.where(tracking.imageNumber.values == x[0])[0][0]
        maxX = np.where(tracking.imageNumber.values == x[-1])[0][0]
        f = inter(tracking.imageNumber.values[minX:maxX])

        imageFinal = []
        distFinal = []
        for i, j in zip(tracking.imageNumber.values[minX:maxX], f):
            if i >= cycle1[0] and i <= cycle1[1]:
                imageFinal.append(i)
                distFinal.append(j)
            elif i >= cycle2[0] and i <= cycle2[1]:
                imageFinal.append(i)
                distFinal.append(j)
                
        with open(path + "/distanceCor.txt", 'a') as f:
            for i, j in zip(imageFinal, distFinal):
                f.write(str(i) + '\t' + str(j) +'\t' +"0" + '\n')
    return True


def updateToml(pathToml="", dest=None, clean=False):

    dic = toml.load(pathToml);
    path = dic["info"]["path"]

    trackingData = pandas.read_csv(path + "Tracking_Result/tracking.txt", sep="\t")

    isClean = False
    if clean:
      isClean = cleanDist(path, trackingData, dic["experiment"]["product1"], dic["experiment"]["product2"])
    if isClean:
      distanceData = pandas.read_csv(path + "distanceCor.txt", sep="\t")
    else:
      distanceData = pandas.read_csv(path + "distance.txt", sep="\t")

    tmp = np.zeros(len(trackingData.imageNumber.values))
    for k, (i, j) in enumerate(zip(trackingData.imageNumber.values, trackingData.id.values)):
        tmpDist = distanceData[(distanceData.imageNumber == i) & (distanceData.id == j)]
        if not tmpDist.empty:
            tmp[k] = (tmpDist.distance.values[0])
        else:
            tmp[k] = (np.nan)
    trackingData["distance"] = tmp

    tracking = {}
    objects = set(trackingData.id.values)
    header = trackingData.columns.values.tolist()
    for i in objects:
        data = trackingData[trackingData["id"] == i]
        tmpDict = {}
        for head in header:
            tmpDict[head] = data[head]
        tracking["Fish_" + str(i)] = tmpDict
    dic["tracking"] = tracking

    if not dest:
        with open(pathToml, "w") as f:
            toml.dump(dic, f)
    else:
        with open(dest + ".toml", "w") as f:
            toml.dump(dic, f)


parser = argparse.ArgumentParser(description="Update the tracking data of a toml file that contains all the information about the experiment")
parser.add_argument("path", nargs='+', help="Path to the toml file")
parser.add_argument("-o", dest="dest", default=None, help="Path and filename of toml file")
parser.add_argument("--clean", dest="clean", action='store_false', help="Clean the distance data by removing artefacts.")

args = parser.parse_args()
for i in args.path:
  if i:
    updateToml(i, args.dest, clean=args.clean)
print("Done")
