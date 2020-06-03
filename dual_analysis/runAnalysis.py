from distribution import *

import numpy as np
import seaborn as sns
import glob
import numpy as np
import pandas
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
from matplotlib.patches import Rectangle

fig, axs = plt.subplots(6, 3, figsize=(10, 10))
fig2, axs2 = plt.subplots(1, 3, figsize=(10, 5))
concentration = [0.01, 0.02, 0.03, 0.06, 0.25, 0.5]
A = []
B = []
C = []
angleDist = []
timeDist = []
lenDist = []
labels = []
PI = []
for i, conc in enumerate(concentration):
    paths = glob.glob("/home/benjamin/Documents/Jupyter_Notebooks/citric_larva/citric_larva/data/Science/Project/Behavior/Dual/Data/Larva/CitricAcid/" + str(conc) + "/*/*/")
    prodT = []
    bufferT = []
    prodL = []
    bufferL = []
    prodC = []
    bufferC = []
    pref = []
    prefh = []
    firstBout = [[], [], [], [], [], []]
    for j in paths:
        #try:
            tracking = pandas.read_csv(j + "/Tracking_Result/tracking.txt", sep='\t')
            for iD in range(np.max(tracking.id.values) + 1):
                tmp = concatenateData(j, iD)
                tmpBouts = extractBouts(tmp)
                a, b = boutTimeDist(inside=tmpBouts["inProd"], outside=tmpBouts["outProd"])
                c, d = boutLenDist(inside=tmpBouts["inProd"], outside=tmpBouts["outProd"])
                e, f = boutAngleDist(inside=tmpBouts["inProd"], outside=tmpBouts["outProd"])
                bufferT.extend(a)
                prodT.extend(b)
                #prefh.append(h)
                bufferL.extend(c)
                prodL.extend(d)
                prodC.extend(f)
                bufferC.extend(e)
                tmpPi = pi(tmp)
                if not np.isnan(tmpPi) and abs(tmpPi) != 1:
                    pref.append(tmpPi)
                
                '''trajects = extractTraj(tmp)
                trajBoutsBuff = []
                trajBoutsProd = [] 
                for ii in trajects["inProd"]:
                    if len(extractBoutsSimple(ii)) > 0:
                        trajBoutsProd.append(extractBoutsSimple(ii)[0])
                    else:
                        plt.figure()
                        plt.plot(ii.xHead.values, ii.yHead.values, 'o-')
                        plt.title(str(ii.imageNumber.values[0]) + j)
                        plt.xlim(0, 1000)
                        plt.ylim(0, 500)
                for ii in trajects["outProd"]:
                    if len(extractBoutsSimple(ii)) > 0:
                        trajBoutsBuff.append(extractBoutsSimple(ii)[0])
                print("A", len(trajBoutsProd), len(trajects["inProd"]))
                h, k = boutTimeDist(inside=trajBoutsBuff, outside=trajBoutsProd)
                l, m = boutLenDist(inside=trajBoutsBuff, outside=trajBoutsProd)
                n, o = boutAngleDist(inside=trajBoutsBuff, outside=trajBoutsProd)
                firstBout[0].extend(h)
                firstBout[1].extend(k)
                firstBout[2].extend(l)
                firstBout[3].extend(m)
                firstBout[4].extend(n)
                firstBout[5].extend(o)
        #except Exception as e:
         #   print("error", i, e)
            
    with open(str(conc) + ".txt", "w") as f:
        f.write("time buffer" + '\t')
        for ind in bufferT:
            f.write(str(ind) + '\t')
        f.write('\n' + "time product" + '\t')
        for ind in prodT:
            f.write(str(ind) + '\t')
        f.write('\n' + "len buffer" + '\t')
        for ind in bufferL:
            f.write(str(ind) + '\t')
        f.write('\n' + "len product" + '\t')
        for ind in prodL:
            f.write(str(ind) + '\t')
        f.write('\n' + "ang buffer" + '\t')
        for ind in bufferC:
            f.write(str(ind) + '\t')
        f.write('\n' + "ang product" + '\t')
        for ind in prodC:
            f.write(str(ind) + '\t')
        f.write('\n' + "pref index" + '\t')
        for ind in pref:
            f.write(str(ind) + '\t')'''
        

    sns.distplot(bufferT, label="Buffer", ax=axs[i, 0], norm_hist=True, kde=False, kde_kws={"gridsize": 1000, "clip": (0, 5)}, bins=np.linspace(0, 10, 100))
    sns.distplot(prodT, label="Product", ax=axs[i, 0], norm_hist=True, kde=False, kde_kws={"gridsize": 1000, "clip": (0, 5)}, bins=np.linspace(0, 10, 100))
    #sns.distplot(firstBout[0], label=" First Buffer", ax=axs[i, 0], norm_hist=True, kde=False, kde_kws={"gridsize": 1000, "clip": (0, 5)}, bins=np.linspace(0, 10, 100))
    #sns.distplot(firstBout[1], label=" First Product", ax=axs[i, 0], norm_hist=True, kde=False, kde_kws={"gridsize": 1000, "clip": (0, 5)}, bins=np.linspace(0, 10, 100))
    axs[i, 0].set_xlim(0, 5)
    axs[i, 0].set_xlabel("Time (seconds)")
    axs[i, 0].legend()
    sns.distplot(bufferL, label="Buffer", ax=axs[i, 1], norm_hist=True, kde=False, kde_kws={"gridsize": 500}, bins=np.linspace(0, 50, 100))
    sns.distplot(prodL, label="Product", ax=axs[i, 1], norm_hist=True, kde=False, kde_kws={"gridsize": 500}, bins=np.linspace(0, 50, 100))
    #sns.distplot(firstBout[2], label=" First Buffer", ax=axs[i, 1], norm_hist=True, kde=False, kde_kws={"gridsize": 1000, "clip": (0, 5)}, bins=np.linspace(0, 50, 100))
    #sns.distplot(firstBout[3], label=" First Product", ax=axs[i, 1], norm_hist=True, kde=False, kde_kws={"gridsize": 1000, "clip": (0, 5)}, bins=np.linspace(0, 50, 100))
    axs[i, 1].set_xlim(0, 25)
    axs[i, 1].set_xlabel("Length (mm)")
    axs[i, 1].legend()
    sns.distplot(bufferC, label="Buffer", ax=axs[i, 2], norm_hist=True, kde=False, kde_kws={"gridsize": 500}, bins=np.linspace(-3, 3, 60))
    sns.distplot(prodC, label="Product", ax=axs[i, 2], norm_hist=True, kde=False, kde_kws={"gridsize": 500}, bins=np.linspace(-3, 3, 60))
    #sns.distplot(firstBout[4], label=" First Buffer", ax=axs[i, 2], norm_hist=True, kde=False, kde_kws={"gridsize": 1000, "clip": (0, 5)}, bins=np.linspace(-3, 3, 60))
    #sns.distplot(firstBout[5], label=" First Product", ax=axs[i, 2], norm_hist=True, kde=False, kde_kws={"gridsize": 1000, "clip": (0, 5)}, bins=np.linspace(-3, 3, 60))
    #axs[i, 2].set_xlim(-3.14, 3.14)
    axs[i, 2].set_xlabel("Angle(rad)")
    axs[i, 2].legend()
    
    
    sns.kdeplot(prodT, label=str(conc), ax=axs2[0])
    A.extend(bufferT)
    sns.kdeplot(prodL, label=str(conc), ax=axs2[1])
    B.extend(bufferL)
    sns.kdeplot(prodC, label=str(conc), ax=axs2[2])
    C.extend(bufferC)
    timeDist.append(prodT)
    lenDist.append(prodL)
    angleDist.append(prodC)
    labels.append(conc)
    PI.append(pref)
    
sns.kdeplot(A, label="control", ax=axs2[0], gridsize=1000, clip=(0,5))
sns.kdeplot(B, label="control", ax=axs2[1], gridsize=500)
sns.kdeplot(C, label="control", ax=axs2[2], gridsize=500)
timeDist.insert(0, A)
lenDist.insert(0, B)
angleDist.insert(0, C)
PI.insert(0, 0)
labels.insert(0, 0)
axs2[0].set_xlim(0, 5)
axs2[0].set_xlabel("Time (seconds)")
axs2[0].legend()
axs2[1].set_xlim(0, 25)
axs2[1].set_xlabel("Length (mm)")
axs2[1].legend()
axs2[2].set_xlim(-3.14, 3.14)
axs2[2].set_yscale('log')
axs2[2].set_xlabel("Angle(rad)")
axs2[2].legend()
fig.savefig("bout_dist.svg")
fig2.savefig("bout_sum.svg")
print(PI)

'''path = "/home/benjamin/Documents/Controls/data/Science/Project/Behavior/Dual/Data/Repulsion/AcideCitrique/0.01/2018-02-28/Run 1.05/"
tracking = concatenateData(path)
print(len(tracking[tracking.cycle == 1].imageNumber.values), len(tracking.imageNumber.values))
print((tracking[tracking.cycle == 1].imageNumber.values[-1] - tracking[tracking.cycle == 1].imageNumber.values[0])/(tracking.imageNumber.values[-1] - tracking.imageNumber.values[0]))
for i, j in enumerate(tracking.imageNumber.values):
    if tracking.cycle.values[i] == 2:
        plt.figure()
        plt.plot(tracking.xHead.values[i-10:i+1], tracking.yHead.values[i-10:i+1])
        if tracking.cross.values[i] == 1:
            plt.plot(tracking.xHead.values[i], tracking.yHead.values[i], 'ro')
        plt.xlim(0,1000)
        plt.ylim(0,500)
        plt.title(str(tracking.imageNumber.values[i]))
        plt.savefig("film_test/frame_" + str(i).zfill(6) + ".png")
        plt.close()'''