import  os
import pandas
import argparse
import glob

def extractTimestamp(path=""):

    milestones = pandas.read_csv(path + "Milestones.txt", sep="\t", header=None)
    timeList = list(milestones[0])

    with open(path + "/TimestampsTest.txt", "w") as f:
        frames = glob.glob("*Frame_*")
        for i, j in enumerate(frames):
            r = open(j, "r")
            imageNumber = int(j[j.find("Frame_"): -4])
            timestamp = r.readlines()[-1][line.find(":") + 1 ::]

            if imageNumber in timeList:
               timestamp = milestones[1][timeList.index(imageNumber)] 
            f.write(imageNumber + '\t' + timestamp + '\n')


parser = argparse.ArgumentParser(description="Extract the timestamp from the images") 
parser.add_argument("path", help="Path to the folder of the experiment")
args = parser.parse_args()
extractTimestamp(args.path)
