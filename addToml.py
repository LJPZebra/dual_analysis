import pandas
import toml
import numpy as np
import argparse
import os
import datetime

def addToml(path="", fileName=None, dest="", keyFile):

    folders = os.path.abspath(path).split(os.sep)
    
    if not fileName:
        fileName = folders[-2].replace("-", "") + folders[-1].replace("Run ", "").replace(".", "")

    fileAdded = pandas.read_csv(path + "distance.txt", sep='\t', header=0)

    dic = toml.load(path + fileName + ".toml");
    interfaces = {"image": fileAdded.imageNumber.values, "distance": fileAdded.distance.values}
    dic["interface"] = interfaces

    if not dest:
        with open(path + fileName + ".toml", "w") as f:
            toml.dump(dic, f)
    else:
        with open(dest + fileName + ".toml", "w") as f:
            toml.dump(dic, f)


parser = argparse.ArgumentParser(description="Update the tracking data of a toml file that contains all the information about the experiment")
parser.add_argument("path", nargs='+', help="Path to the folder of the experiment")
parser.add_argument("--keyFile", dest="keyFile", help="File associated to key to add")
parser.add_argument("--name", dest="name", help="Name of the output toml file")
parser.add_argument("-o", dest="dest", help="Path to a folder to store the toml file")

args = parser.parse_args()
for i in args.path:
  if i:
    updateToml(i, args.name, args.dest, args.keyFile)
print("Done")
