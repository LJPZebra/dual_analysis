import os
import pandas
import argparse
import glob
import subprocess

def extractTimestamp(path, fmt):


    milestones = pandas.read_csv(path + "/Milestones.txt", sep="\t", header=None)
    timeList = list(milestones[0])

    with open(path + "/Timestamps.txt", "w") as f:
        frames = glob.glob(path + "*Frame_*")
        frames.sort()
        for i, j in enumerate(frames):
          if fmt == "png":
            imageNumber = int(j[j.find("Frame_") + 6: -4])
            process = subprocess.Popen(["/usr/bin/exiftool", j],stdout=subprocess.PIPE, stderr=subprocess.STDOUT,universal_newlines=True)
            for tag in process.stdout:
                  line = tag.strip().split(':')
                  if line[0].strip() == "Description":
                    line[-1] = line[-1].strip()
                    place = line[-1].find(" ")
                    timestamp = line[-1][place + 1 ::]
                    break
          else :
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
parser.add_argument("--format", dest="format", help="Image format")
args = parser.parse_args()
print(args.path)
for i in args.path:
  extractTimestamp(i, args.format)
  print("Done " + i)

