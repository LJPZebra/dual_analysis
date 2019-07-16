import os
import pandas
import argparse
import glob

def extractTimestamp(path=""):

    milestones = pandas.read_csv(path + "/Milestones.txt", sep="\t", header=None)
    timeList = list(milestones[0])

    with open(path + "/Timestamps.txt", "w") as f:
        frames = glob.glob(path + "*Frame_*")
        frames.sort()
        for i, j in enumerate(frames):
            with open(j, encoding="utf8", errors="ignore") as r:
              imageNumber = int(j[j.find("Frame_") + 6: -4])
              line = r.readlines()[-1]
              place = line.find(":")
              timestamp = line[place + 1 ::]

            if imageNumber in timeList:
               timestamp = milestones[1][timeList.index(imageNumber)] 
            f.write(str(imageNumber) + '\t' + str(timestamp) + '\n')


parser = argparse.ArgumentParser(description="Extract the timestamp from the images") 
parser.add_argument("path", nargs='+', help="Path to the folder of the experiment")
args = parser.parse_args()
print(args.path)
for i in args.path:
  extractTimestamp(i)
  print("Done " + i)

