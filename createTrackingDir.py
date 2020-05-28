import os
import numpy as np
import pandas
import argparse
import glob
import subprocess


def createDir(path):
    """
    Create a Tracking_Result folder with a standard tracking.txt file from an older tracking.txt file.
    """

    # Create the Tracking_Result folder 
    if not os.path.exists(path + "/Tracking_Result"):
      os.mkdir(path + "/Tracking_Result")

    # Open the dual experiment files  
    inputFile = pandas.read_csv(path + "/tracking.txt", sep="   ", header=1, skiprows=0, engine="python")
    timeStamp = pandas.read_csv(path + "/Timestamps.txt", sep="\t", header=None)
    milestones = pandas.read_csv(path + "/Milestones.txt", sep="\t", header=None)


    # Replace the timestamp with the image number
    for i, j in enumerate(inputFile["imageNumber"].values):
      for k, l  in enumerate(timeStamp[1]):
        if str(j) == str(l):
          inputFile["imageNumber"][i] = timeStamp[0][k]
          break
# Ordre des boucles important car y a 2 buffers !!!!
    for k, l  in enumerate(milestones[2]):
      for i, j in enumerate(inputFile["imageNumber"]):
        if str(j) == "#" + str(l):
          inputFile["imageNumber"][i] = milestones[0][k]
          #break

    # Create the id column in the tracking.txt file
    idList = [0]
    for i, j in enumerate(inputFile["imageNumber"].values):
      if i > 0:
        if inputFile["imageNumber"][i-1] != inputFile["imageNumber"][i]:
          idList.append(0)
        else:
          idList.append(idList[-1]+1)
          #print("error", i)
    inputFile["id"] = idList
    inputFile = inputFile.drop(inputFile[inputFile.id != 0].index) # Clean the surnumeral object. To comment if there is several objects in the esperiment.

    inputFile.reset_index(inplace=True)
    dropList = []
    meanLen = np.mean(inputFile.lenght) + 100
    maxId = np.max(inputFile.id.values)
    for i, __ in enumerate(inputFile.imageNumber.values):
      if i < len(inputFile.imageNumber.values) - maxId - 1:
        if inputFile.iloc[i].xHead == inputFile.iloc[i+maxId+1].xHead and inputFile.iloc[i].yBody == inputFile.iloc[i+maxId+1].yBody:
          dropList.append(i+maxId+1)
    #  if int(inputFile.lenght.iloc[i]) > meanLen + 100 or int(inputFile.lenght.iloc[i]) < 50:
     #   dropList.append(i)

    #inputFile.drop(dropList, axis=0, inplace=True)

    inputFile.to_csv(path + "/Tracking_Result/tracking.txt", sep="\t", index=None)





parser = argparse.ArgumentParser(description="Create the tracking directory with a standard tracking.txt file  for very old Fishy Tracking results") 
parser.add_argument("path", nargs='+', help="Path to the folder of the experiment")

args = parser.parse_args()
log = []
for i in args.path:
  try:
    if os.path.exists(i + "/tracking.txt"):
      createDir(i)
      print("Done ", i)
  except Exception as e:
    print(e)
    log.append(i)
print("These files can not be converted \n ")
print(log)
