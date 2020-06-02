import toml
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

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

def activity(experiment):

  indexRef = experiment["metadata"]["image"].index(0)
  time = np.array(experiment["metadata"]["time"]) - experiment["metadata"]["time"][indexRef]
  activity = []

  for key in experiment["tracking"]:
    image = experiment["tracking"][key]["imageNumber"]
    image = [int(i) for i in image]
    dtimeFish = np.diff(time[image])*1e-9
    dtimeFish = np.insert(dtimeFish, 0, 0)
    dlFish = np.sqrt(np.diff(experiment["tracking"][key]["xHead"])**2 + np.diff(experiment["tracking"][key]["yHead"])**2)
    dlFish = np.insert(dlFish, 0, 0)
    protocol = cycle(experiment, key)

    act = []
    for i in protocol:
      act.append(np.sum(dlFish[i]))

    #act = [i/(act[0]) for i in act]

    activity.append(act)

  experiment["experiment"]["activity"] = activity

  return activity


def preferenceIndex(experiment):

# Get the starting time and the absolute time of the experiment
  indexRef = experiment["metadata"]["image"].index(0)
  time = np.array(experiment["metadata"]["time"]) - experiment["metadata"]["time"][indexRef]

# For each fish in an experiment
  pref = []
  xDist = []
  if(len(experiment["tracking"]) != 1 ): print("Multiple fish detected")
  for key in experiment["tracking"]:

# Separate cycles
    image = experiment["tracking"][key]["imageNumber"]
    image = [int(i) for i in image]
    #dtimeFish = np.zeros(len(image))
    #for i, j in enumerate(image):
    #  dtimeFish[i] = time[experiment["metadata"]["image"].index(j)]
    dtimeFish = np.diff(time[image])

    dtimeFish = np.insert(dtimeFish, 0, 0)
    position = np.array(experiment["tracking"][key]["xHead"])
    interfacePosition = float(experiment["experiment"]["interface"])

    protocol = cycle(experiment, key)

    preferenceIndex = []
    for i, j in enumerate(protocol):
      if len(j) != 0 :
        left = np.float64(np.sum(dtimeFish[j][np.intersect1d(np.where(position[j] < interfacePosition)[0], np.where(dtimeFish[j] < 5*60707805))]))
        right = np.float64(np.sum(dtimeFish[j][np.intersect1d(np.where(position[j] > interfacePosition)[0], np.where(dtimeFish[j] < 5*60707805))]))
        preferenceIndex.append( (left - right) / (left + right) )
      else:
        preferenceIndex.append( np.nan )
      if len(j) < 10: 
        print("Warning low stat" + experiment["info"]["path"] + str(len(experiment["tracking"])))
        #preferenceIndex[-1] =  np.nan

    order  = experiment["experiment"]["order"]
    if order == "BLBR":
      preferenceIndex[3] = - preferenceIndex[3]
      preferenceIndex[2] = - preferenceIndex[2]
    elif order == "BRBL":
      preferenceIndex[0] = - preferenceIndex[0]
      preferenceIndex[1] = - preferenceIndex[1]
    else:
      raise ValueError("Experiment order is empty.")

    pref.append(preferenceIndex)
    xDist.append(position[protocol[0]])

  experiment["experiment"]["preference index"] = pref
  experiment["experiment"]["x dist"] = xDist
  return pref


def plot(experiments, savePath):

  concentrations = []
  products = []
  for i in experiments:
    concentrations.append(float(i["experiment"]["concentration"]))
    products.append(i["experiment"]["product"])
  concentrations = list(set(concentrations))
  concentrations.sort()
  products = list(set(products))
  print(concentrations)

  piConcentrationBuf1 = [ [] for _ in range(len(concentrations))]
  piConcentrationProd1 = [ [] for _ in range(len(concentrations))]
  piConcentrationBuf2 = [ [] for _ in range(len(concentrations))]
  piConcentrationProd2 = [ [] for _ in range(len(concentrations))]
  piActivityBuf1 = [ [] for _ in range(len(concentrations))]
  piActivityProd1 = [ [] for _ in range(len(concentrations))]
  piActivityBuf2 = [ [] for _ in range(len(concentrations))]
  piActivityProd2 = [ [] for _ in range(len(concentrations))]


  distProd = [ [ [] for _ in range(len(concentrations))],  [ [] for _ in range(len(concentrations))], [ [] for _ in range(len(concentrations))], [ [] for _ in range(len(concentrations))] ]
  distBuff = [ [ [] for _ in range(len(concentrations))],  [ [] for _ in range(len(concentrations))], [ [] for _ in range(len(concentrations))], [ [] for _ in range(len(concentrations))] ]

  xDist = []
  xDistDual = [ [], [], [], [] ]



  for exp in experiments:
    for i, j in enumerate(products):
      for k, l in enumerate(concentrations):
        if exp["experiment"]["product"] == j  and exp["experiment"]["concentration"] == l:
          for m, n in enumerate(exp["experiment"]["preference index"]):
            piConcentrationProd1[k].append(n[1]) if(abs(n[1]) < 0.95) else piConcentrationProd1[k].append(np.nan)
            piConcentrationProd2[k].append(n[3]) if(abs(n[3]) < 0.95) else piConcentrationProd2[k].append(np.nan)
            piConcentrationBuf1[k].append(n[0]) if(abs(n[0]) < 0.95) else piConcentrationBuf1[k].append(np.nan)
            piConcentrationBuf2[k].append(n[2]) if(abs(n[2]) < 0.95) else piConcentrationBuf2[k].append(np.nan)
          for m, n in enumerate(exp["experiment"]["activity"]):
            piActivityProd1[k].append(n[1])
            piActivityProd2[k].append(n[3])
            piActivityBuf1[k].append(n[0])
            piActivityBuf2[k].append(n[2])

          for i in exp["experiment"]["x dist"]:
            if i.size:
              i -= np.min(i)
              xDist.extend(i)
              xDistDual[int(exp["info"]["path"][-5:-4]) - 1].extend(i)

  for i, _ in enumerate(concentrations):
    fig, ax = plt.subplots()
    ax.set_title(str(concentrations[i]))
    for j, _ in enumerate(piConcentrationBuf1[i]):
      ax.plot(np.random.rand(4)+ np.arange(4*2, step=2), [piConcentrationBuf1[i][j], piConcentrationProd1[i][j], piConcentrationBuf2[i][j], piConcentrationProd2[i][j]], '-o', lw=0.5)




  piConcentrationProd1 = [np.asarray(i) for i in piConcentrationProd1]
  piConcentrationProd1 = [i[~np.isnan(i)] for i in piConcentrationProd1]
  piConcentrationBuf1 = [np.asarray(i) for i in piConcentrationBuf1]
  piConcentrationBuf1 = [i[~np.isnan(i)] for i in piConcentrationBuf1]
  piConcentrationProd2 = [np.asarray(i) for i in piConcentrationProd2]
  piConcentrationProd2 = [i[~np.isnan(i)] for i in piConcentrationProd2]
  piConcentrationBuf2 = [np.asarray(i) for i in piConcentrationBuf2]
  piConcentrationBuf2 = [i[~np.isnan(i)] for i in piConcentrationBuf2]

  piActivityProd1 = [np.asarray(i) for i in piActivityProd1]
  piActivityProd1 = [i[~np.isnan(i)] for i in piActivityProd1]
  piActivityBuf1 = [np.asarray(i) for i in piActivityBuf1]
  piActivityBuf1 = [i[~np.isnan(i)] for i in piActivityBuf1]
  piActivityProd2 = [np.asarray(i) for i in piActivityProd2]
  piActivityProd2 = [i[~np.isnan(i)] for i in piActivityProd2]
  piActivityBuf2 = [np.asarray(i) for i in piActivityBuf2]
  piActivityBuf2 = [i[~np.isnan(i)] for i in piActivityBuf2]

  fig, axs = plt.subplots(4, 1, sharex=True, sharey=True, figsize=(11.69, 8.27))
  boxWidthCompensation = lambda p, w: 10**(np.log10(p)+w/2.)-10**(np.log10(p)-w/2.)
  widths = [boxWidthCompensation(p, 0.05) for p in concentrations]

  for i, j in enumerate(piConcentrationProd1):
    axs[1].scatter(np.ones(len(j))*(concentrations[i]), j, s=8)
  axs[1].boxplot(piConcentrationProd1, positions=concentrations, widths= widths)

  for i, j in enumerate(piConcentrationProd2):
    axs[3].scatter(np.ones(len(j))*(concentrations[i]), j, s=8)
  axs[3].boxplot(piConcentrationProd2, positions=concentrations, widths= widths)

  axs[0].boxplot(piConcentrationBuf1, positions=concentrations, widths= widths)
  for k, l in enumerate(piConcentrationBuf1):
    axs[0].scatter(np.ones(len(l))*(concentrations[k]), l, s=8)
    axs[0].annotate(str('N = '  + str(len(l))), xy=(concentrations[k], 1.2),  xycoords='data', horizontalalignment='center', verticalalignment='top', size=10,)

  axs[2].boxplot(piConcentrationBuf2, positions=concentrations, widths= widths)
  for k, l in enumerate(piConcentrationBuf2):
    axs[2].scatter(np.ones(len(l))*(concentrations[k]), l, s=8)
  

  axs[1].plot(concentrations, [np.mean(i) for i in piConcentrationProd1], color="black")
  axs[2].plot(concentrations, [np.mean(i) for i in piConcentrationBuf2], color="black")
  axs[0].plot(concentrations, [np.mean(i) for i in piConcentrationBuf1], color="black")
  axs[3].plot(concentrations, [np.mean(i) for i in piConcentrationProd2], color="black")
  
  axs[0].set_xscale("log")
  for i in axs:
    i.grid(b=True)

  axs[0].set_ylim(-1.2, 1.2)
  axs[0].set_xlim(0, concentrations[-1] + widths[-1]*1.5)
  axs[1].set_ylabel("Product Cycle #1")
  axs[0].set_ylabel("Buffer Cycle #1")
  axs[3].set_ylabel("Product Cycle #2")
  axs[2].set_ylabel("Buffer Cycle #2")

  if savePath:
    path = os.path.abspath(savePath) + "/"
    if not os.path.exists(path):
          os.makedirs(path)
    plt.savefig(path + "/" + "preferenceIndex.svg")

  fig, ax = plt.subplots()
  meanProd = []
  stdProd = []
  for i, j in enumerate(concentrations):
    tmp = np.concatenate((piConcentrationProd1[i], piConcentrationProd2[i]))
    meanProd.append(float(np.mean(tmp)))
    stdProd.append(float(np.std(tmp)))
  output = {'mean': meanProd, 'std' : stdProd, 'concentration' : concentrations}
  with open("/home/benjamin/Documents/preferenceIndexPlots/quinine_juvenil.toml", "w") as f:
    f.write(toml.dumps(output))

  fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(11.69, 8.27))
  boxWidthCompensation = lambda p, w: 10**(np.log10(p)+w/2.)-10**(np.log10(p)-w/2.)
  widths = [boxWidthCompensation(p, 0.05) for p in concentrations]

  piBuffer =  []
  piProd = []
  for i, j in enumerate(concentrations):
    piBuffer.append(np.concatenate((piConcentrationBuf1[i], piConcentrationBuf2[i])))
    piProd.append(np.concatenate((piConcentrationProd1[i], piConcentrationProd2[i])))

  for i, j in enumerate(piBuffer):
    axs[0].scatter(np.ones(len(j))*(concentrations[i]), j, s=8)
  axs[0].boxplot(piBuffer, positions=concentrations, widths= widths)
  for i, j in enumerate(piProd):
    axs[1].scatter(np.ones(len(j))*(concentrations[i]), j, s=8)
  axs[1].boxplot(piProd, positions=concentrations, widths= widths)

  axs[0].set_xscale("log")
  for i in axs:
    i.grid(b=True)

  axs[0].set_ylim(-1.2, 1.2)
  axs[0].set_xlim(0, concentrations[-1] + widths[-1]*1.5)
  axs[1].set_ylabel("Product Cycle")
  axs[0].set_ylabel("Buffer Cycle")

  fig, axs = plt.subplots(4, 1, sharex=True, sharey=True)
  for i, j in enumerate(piActivityProd1):
    axs[1].scatter(np.ones(len(j))*(concentrations[i]), j, s=8)
  axs[1].boxplot(piActivityProd1, positions=concentrations, widths= widths)

  for i, j in enumerate(piActivityProd2):
    axs[3].scatter(np.ones(len(j))*(concentrations[i]), j, s=8)
  axs[3].boxplot(piActivityProd2, positions=concentrations, widths= widths)

  axs[0].boxplot(piActivityBuf1, positions=concentrations, widths= widths)
  for k, l in enumerate(piActivityBuf1):
    axs[0].scatter(np.ones(len(l))*(concentrations[k]), l, s=8)

  axs[2].boxplot(piActivityBuf2, positions=concentrations, widths= widths)
  for k, l in enumerate(piActivityBuf2):
    axs[2].scatter(np.ones(len(l))*(concentrations[k]), l, s=8)
  
  axs[1].plot(concentrations, [np.mean(i) for i in piActivityProd1], color="black")
  axs[2].plot(concentrations, [np.mean(i) for i in piActivityBuf2], color="black")
  axs[0].plot(concentrations, [np.mean(i) for i in piActivityBuf1], color="black")
  axs[3].plot(concentrations, [np.mean(i) for i in piActivityProd2], color="black")
  
  axs[1].set_ylabel("Product1")
  axs[0].set_ylabel("Buffer1")
  axs[3].set_ylabel("Product2")
  axs[2].set_ylabel("Buffer2")

  axs[0].set_xscale("log")
  for i in axs:
    i.grid(b=True)

  plt.figure()
  sns.distplot(np.array(xDist))

  for j, i in enumerate(xDistDual):
    if i:
      plt.figure()
      sns.distplot(np.array(i))
      plt.title("Dual " + str(j + 1))
  


  '''fig, axs = plt.subplots(4, 1, sharex=True, sharey=True)
  for i in range(4):
    axs[i].errorbar(concentrations, [np.mean(distBuff[i][j]) for j in range(len(concentrations))], yerr=[np.std(distBuff[i][j]) for j in range(len(concentrations))], label="Buffer")
    axs[i].errorbar(concentrations, [np.mean(distProd[i][j]) for j in range(len(concentrations))], yerr=[np.std(distProd[i][j]) for j in range(len(concentrations))], label="Product")
    #axs[i].errorbar(concentrations, np.mean(distProd[i]), yerr=np.std(distProd[i]))
  axs[0].set_xscale("log")
  axs[0].legend();


  fig, ax = plt.subplots();
  c, p, a = [], [], []
  for exp in experiments:
    for m, n in enumerate(exp["experiment"]["preference index"]):
        c.append(exp["experiment"]["concentration"])
        p.append(n[1])
        a.append(exp["fish"]["age"])
  ax.scatter(c, p, a)'''




  plt.show()

parser = argparse.ArgumentParser(description="Compute the preference index from a list of toml files")
parser.add_argument("path", nargs='+', help="List of path")
parser.add_argument("--write", dest="write", help="Path to a folder where to write the output")

args = parser.parse_args()
a= getTomlFile(args.path)
for i in a:
  print(preferenceIndex(i))
  activity(i)
plot(a, args.write)
