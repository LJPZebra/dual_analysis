import pandas
import toml
import numpy as np
import argparse
import os
import datetime

def updateToml(path="", fileName=None, dest =""):

    folders = os.path.abspath(path).split(os.sep)
    
    if not fileName:
        fileName = folders[-2].replace("-", "") + folders[-1].replace("Run ", "").replace(".", "")


    trackingData = pandas.read_csv(path + "Tracking_Result/tracking.txt", sep="\t")

    tracking = {}
    objectNumber = np.max(trackingData["id"]) + 1
    print(objectNumber)
    header = trackingData.columns.values.tolist()
    for i in range(objectNumber):
        data = trackingData[trackingData["id"] == i]
        tmpDict = {}
        for head in header:
            tmpDict[head] = data[head]
        tracking["Fish_" + str(i)] = tmpDict





    if not dest:
        dic = toml.load(path + fileName + ".toml");
        dic["tracking"] = tracking
        with open(path + fileName + ".toml", "w") as f:
            toml.dump(dic, f)
    else:
        dic = toml.load(dest + fileName + ".toml");
        dic["tracking"] = tracking
        with open(dest + fileName + ".toml", "w") as f:
            toml.dump(dic, f)


parser = argparse.ArgumentParser(description="Update the tracking data of a toml file that contains all the information about the experiment")
parser.add_argument("path", nargs='+', help="Path to the folder of the experiment")
parser.add_argument("--name", dest="name", help="Name of the output toml file")
parser.add_argument("-o", dest="dest", help="Path to a folder to store the toml file")

args = parser.parse_args()
for i in args.path:
  if i:
    updateToml(i, args.name, args.dest)
print("Done")
