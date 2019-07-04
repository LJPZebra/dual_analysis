import os
import argparse

def listPath(path):
    pathList = '"'
    for root, __, files in os.walk(path):
        if files and files[0][0:5] == "Frame":
            pathList += str(os.path.abspath(root) + os.sep) + ","
    pathList += '"'

    return pathList

parser = argparse.ArgumentParser(description="Return the path to all the Dual experiment inside a root path") 
parser.add_argument("path", help="Root path where to search for Dual experiment folders")
args = parser.parse_args()
print(listPath(args.path))
