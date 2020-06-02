import toml
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def getTomlFile(pathList):
  files = []
  for path in pathList:
    files.append(toml.load(path))
  return files

def cycle(experiment, fish):
    image = np.array(experiment["tracking"][fish]["imageNumber"])
    protocol = [ ]
    protocol.append(np.intersect1d(np.where(image >= experiment["experiment"]["buffer1"][0]), np.where(image <= experiment["experiment"]["buffer1"][1])))
    protocol.append(np.intersect1d(np.where(image >= experiment["experiment"]["product1"][0]), np.where(image <= experiment["experiment"]["product1"][1])))
    protocol.append(np.intersect1d(np.where(image >= experiment["experiment"]["buffer2"][0]), np.where(image <= experiment["experiment"]["buffer2"][1])))
    protocol.append(np.intersect1d(np.where(image >= experiment["experiment"]["product2"][0]), np.where(image <= experiment["experiment"]["product2"][1])))
    return protocol


def trace(experiment, savePath):

# Get the starting time and the absolute time of the experiment
  indexRef = experiment["metadata"]["image"].index(0)
  time = (np.array(experiment["metadata"]["time"]) - experiment["metadata"]["time"][indexRef])*1e-9


  for key in experiment["tracking"]:
    image = experiment["tracking"][key]["imageNumber"]
    print(experiment["info"]["path"])
    image = [int(i) for i in image]

    position = np.array(experiment["tracking"][key]["xHead"])
    interfacePosition = int(experiment["experiment"]["interface"])

    protocol = cycle(experiment, key)

    fig, axs = plt.subplots(4, 1, sharey=True)
    for i, j in enumerate(protocol):
      if not j.size == 0:
        axs[i].plot(time[j], position[j])
        axs[i].set_xlim(time[j][0], time[j][-1])
        axs[i].set_ylim(-200, 1200)

    axs[0].set_title("Concentration: " + str(experiment["experiment"]["concentration"]) + str(experiment["info"]["path"]) + str(key)[-1] )

    order  = experiment["experiment"]["order"]
    interface  = float(experiment["experiment"]["interface"])
    if order == "BLBR":
      if protocol[0].size != 0:
        axs[0].add_patch(Rectangle((time[protocol[0]][0], 0), time[protocol[0]][-1] - time[protocol[0]][0], 500, color='blue', alpha=0.2))
      if protocol[1].size != 0:
        axs[1].add_patch(Rectangle((time[protocol[1]][0], 0), time[protocol[1]][-1] - time[protocol[1]][0], 500, color='red', alpha=0.2))
      if protocol[2].size != 0:
        axs[2].add_patch(Rectangle((time[protocol[2]][0], interface), time[protocol[2]][-1] - time[protocol[2]][0], 500, color='blue', alpha=0.2))
      if protocol[3].size != 0:
        axs[3].add_patch(Rectangle((time[protocol[3]][0], interface), time[protocol[3]][-1] - time[protocol[3]][0], 500, color='red', alpha=0.2))
    elif order == "BRBL":
      if protocol[0].size != 0:
        axs[0].add_patch(Rectangle((time[protocol[0]][0], interface), time[protocol[0]][-1] - time[protocol[0]][0], 500, color='blue', alpha=0.2))
      if protocol[1].size != 0:
        axs[1].add_patch(Rectangle((time[protocol[1]][0], interface), time[protocol[1]][-1] - time[protocol[1]][0], 500, color='red', alpha=0.2))
      if protocol[2].size != 0:
        axs[2].add_patch(Rectangle((time[protocol[2]][0], 0), time[protocol[2]][-1] - time[protocol[2]][0], 500, color='blue', alpha=0.2))
      if protocol[3].size != 0:
        axs[3].add_patch(Rectangle((time[protocol[3]][0], 0), time[protocol[3]][-1] - time[protocol[3]][0], 500, color='red', alpha=0.2))
    else:
      raise ValueError("Experiment order is empty.")


    if savePath:
      path = os.path.abspath(savePath) + "/" + str(experiment["experiment"]["product"]) + "/" + str(experiment["experiment"]["concentration"]) + "/"
      if not os.path.exists(path):
            os.makedirs(path)
      plt.savefig(path + str(experiment["info"]["title"]) + str(key)[-1] + ".svg")



parser = argparse.ArgumentParser(description="Plot the trace from a list of toml files")
parser.add_argument("path", nargs='+', help="List of path")
parser.add_argument("--write", dest="write", help="Path to a folder where to write the output")

args = parser.parse_args()
a = getTomlFile(args.path)
for i in a:
  trace(i, args.write)

plt.show()
