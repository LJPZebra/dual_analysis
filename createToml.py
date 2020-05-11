import pandas
import toml
import numpy as np
import argparse
import os
import datetime

def createToml(path="", fileName=None, dest ="", erase=None):
    """
    Create the toml file from a list of path
    """

    folders = os.path.abspath(path).split(os.sep)
    
    if not fileName:
        fileName = folders[-2].replace("-", "") + folders[-1].replace("Run ", "").replace(".", "")

    if not dest:
        if os.path.isfile(path + fileName + ".toml"):
            return
    else:
        if os.path.isfile(dest + fileName + ".toml"):
            return

    milestones = pandas.read_csv(path + "Milestones.txt", sep="\t", header=None)
    parameters = pandas.read_csv(path + "Parameters.txt", sep="\t", header=None)
    timestamps = pandas.read_csv(path + "Timestamps.txt", sep="\t", header=None)
    trackingData = pandas.read_csv(path + "Tracking_Result/tracking.txt", sep="\t")

    info = { "title" : fileName, "path" : path, "author" : "Benjamin Gallois" }
    fish = { "age" : int(parameters[1][1]), "date" : datetime.datetime.strptime(parameters[1][0], "%Y-%m-%d"),  "type" : "wt", "remark" : ""}
    experiment = {"date" : datetime.datetime.strptime(folders[-2], "%Y-%m-%d"), "product" : parameters[1][2].lower(), "concentration" : float(parameters[1][3]), "interface" : "500", "order" : "BRBL", "buffer1" : [int(milestones[0][1]), int(milestones[0][2])], "buffer2" : [int(milestones[0][5]), int(milestones[0][6])], "product1" : [int(milestones[0][3]), int(milestones[0][4])], "product2" : [int(milestones[0][7]), int(timestamps[0].tail(1))]}
    metadata = { "image" : timestamps[0], "time" : timestamps[1] }

    tracking = {}
    objectNumber = np.max(trackingData["id"]) + 1
    header = trackingData.columns.values.tolist()
    for i in range(objectNumber):
        data = trackingData[trackingData["id"] == i]
        tmpDict = {}
        for head in header:
            tmpDict[head] = data[head]
        tracking["Fish_" + str(i)] = tmpDict


    dic = {"info" : info, "fish" : fish, "experiment" : experiment, "metadata" : metadata, "tracking" : tracking}

    if not dest:
        with open(path + fileName + ".toml", "w") as f:
            toml.dump(dic, f)
    else:
        with open(dest + fileName + ".toml", "w") as f:
            toml.dump(dic, f)


parser = argparse.ArgumentParser(description="Create a toml file that contains all the information about an experiment")
parser.add_argument("path", nargs='+', help="Path to the folder of the experiments")
parser.add_argument("--name", dest="name", help="Name of the output toml file")
parser.add_argument("-o", dest="dest", help="Path to a folder to store the toml file")
parser.add_argument("--erase", dest="erase", help="Erase the toml file")

args = parser.parse_args()
for i in args.path:
  if i:
    try:
      createToml(i, args.name, args.dest, args.erase)
    except Exception as e:
      print("Failed " + i)
      print(e)
print("Done")
