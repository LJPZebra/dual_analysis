import os
import pandas
import argparse
import glob
import subprocess


def createDir(path):
    if not os.path.exists(path + "Tracking_Result"):
      os.mkdir(path + "Tracking_Result")
    inputFile = pandas.read_csv(path + "/tracking.txt", sep="   ", header=1, skiprows=0, engine="python")
    timeStamp = pandas.read_csv(path + "/Timestamps.txt", sep="\t", header=None)
    milestones = pandas.read_csv(path + "/Milestones.txt", sep="\t", header=None)

    for i, j in enumerate(inputFile["imageNumber"]):
      for k, l  in enumerate(timeStamp[1]):
        if str(j) == str(l):
          inputFile["imageNumber"][i] = timeStamp[0][k]
          break
    for i, j in enumerate(inputFile["imageNumber"]):
      for k, l  in enumerate(milestones[2]):
        if str(j) == "#" + str(l):
          inputFile["imageNumber"][i] = timeStamp[0][k]
          break


    idList = [0]
    for i, j in enumerate(inputFile["imageNumber"]):
      if i > 0:
        if inputFile["imageNumber"][i-1] != inputFile["imageNumber"][i]:
          idList.append(0)
        else:
          idList.append(idList[-1]+1)
          print("error")
    inputFile["id"] = idList
    inputFile = inputFile.drop(inputFile[inputFile.id != 0].index)

    inputFile.to_csv(path + "Tracking_Result/tracking.txt", sep="\t", index=None)

parser = argparse.ArgumentParser(description="Creates the tracking dir for very old Fishy Tracking result") 
parser.add_argument("path", nargs='+', help="Path to the folder of the experiment")

args = parser.parse_args()
log = []
for i in args.path:
  try:
    createDir(i)
  except:
    log.append(i)
print(log)
