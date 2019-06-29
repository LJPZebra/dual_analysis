import pandas
import toml
import numpy as np
import argparse
import os

def createToml(path="", fileName=None):

    if not fileName:
        fileName = os.path.basename(os.path.normpath(path))

    milestones = pandas.read_csv(path + "Milestones.txt", sep="\t", header=None)
    parameters = pandas.read_csv(path + "Parameters.txt", sep="\t", header=None)
    timestamps = pandas.read_csv(path + "Timestamps.txt", sep="\t", header=None)
    trackingData = pandas.read_csv(path + "Tracking_Result/tracking.txt", sep="\t")

    info = { "title" : fileName, "path" : path, "author" : "Benjamin Gallois" }
    fish = { "age" : int(parameters[1][1]), "type" : "wt", "remark" : ""}
    experiment = { "product" : parameters[1][2].lower(), "concentration" : float("".join([i for i in parameters[1][3] if i.isdigit()])), "order" : "", "buffer1" : [int(milestones[0][1]), int(milestones[0][2])], "buffer2" : [int(milestones[0][5]), int(milestones[0][6])], "product1" : [int(milestones[0][3]), int(milestones[0][4])], "product2" : [int(milestones[0][7]), int(timestamps[0].tail(1))]}
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

    with open(path + fileName + ".toml", "w") as f:
        toml.dump(dic, f)


parser = argparse.ArgumentParser(description="Create a toml file that contains all the information about the experiment")
parser.add_argument("path", help="Path to the folder of the experiment")
parser.add_argument("--name", dest="name", help="Name of the output toml file")

args = parser.parse_args()
args.path = args.path.replace("[", "").replace("]", "").replace("'", "").split(",")
for i in args.path:
    createToml(i, args.name)
print("Done")
