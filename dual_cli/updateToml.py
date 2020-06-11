import pandas
import toml
import numpy as np
import argparse
import os
import datetime

def updateToml(pathToml="", dest=None):

    dic = toml.load(pathToml);
    path = dic["info"]["path"]

    trackingData = pandas.read_csv(path + "Tracking_Result/tracking.txt", sep="\t")
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
    objectNumber = np.max(trackingData["id"]) + 1
    header = trackingData.columns.values.tolist()
    for i in range(objectNumber):
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

args = parser.parse_args()
for i in args.path:
  if i:
    updateToml(i, args.dest)
print("Done")
