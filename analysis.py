import toml
import numpy as np
import argparse

def getTomlFile(pathList):
  files = []
  for path in pathList:
    files.append(toml.load(path))
  return files

def preferenceIndex(experiment, pooled=True):

# Get the starting time
  indexRef = experiment["metadata"]["image"].index(0)
  time = np.array(experiment["metadata"]["image"]) - experiment["metadata"]["time"][indexRef]
  pref = []

  for key in experiment["tracking"]:

# Separate cycles
    image = np.array(experiment["tracking"][key]["imageNumber"])
    dtime = np.diff(time)
    dtime = np.insert(dtime, 0, 0)
    position = np.array(experiment["tracking"][key]["xHead"])
    interfacePosition = int(experiment["experiment"]["interface"])

    protocol = [ ]
    protocol.append(np.intersect1d(np.where(image >= experiment["experiment"]["buffer1"][0]), np.where(image <= experiment["experiment"]["buffer1"][1])))
    protocol.append(np.intersect1d(np.where(image >= experiment["experiment"]["product1"][0]), np.where(image <= experiment["experiment"]["product1"][1])))
    protocol.append(np.intersect1d(np.where(image >= experiment["experiment"]["buffer2"][0]), np.where(image <= experiment["experiment"]["buffer2"][1])))
    protocol.append(np.intersect1d(np.where(image >= experiment["experiment"]["product2"][0]), np.where(image <= experiment["experiment"]["product2"][1])))

    preferenceIndex = []
    for i, j in enumerate(protocol):
      left = np.sum(dtime[j][np.where(position[j] < interfacePosition)[0]])
      right = np.sum(dtime[j][np.where(position[j] > interfacePosition)[0]])
      preferenceIndex.append( (left - right) / (left + right) )

      
    order  = experiment["experiment"]["order"]
    if order == "BLBR":
      preferenceIndex[3] = - preferenceIndex[3]
      preferenceIndex[2] = - preferenceIndex[2]
    else:
      preferenceIndex[0] = - preferenceIndex[0]
      preferenceIndex[1] = - preferenceIndex[1]

    if pooled:
      pref.append([(preferenceIndex[0] + preferenceIndex[2])*0.5, (preferenceIndex[1] + preferenceIndex[3])*0.5])
    else:
      pref.append(preferenceIndex)

  return pref


parser = argparse.ArgumentParser(description="Compute the preference index from a list of toml files")
parser.add_argument("path", nargs='+', help="List of path")

args = parser.parse_args()
a= getTomlFile(args.path)
for i in a:
  p = preferenceIndex(i, pooled=True)
  print(p)
