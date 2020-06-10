from distribution import *
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


def distribution(path, outPath):
    
    if not outPath:
        outPath = os.path.split(path[0])[0]

    ### Sorting data
    concentrations = set()
    data = []
    for i, j in enumerate(path):
        dic = toml.load(j)
        concentrations.add(dic["experiment"]["concentration"])
        for k, __ in enumerate(dic["tracking"]):
            tracking = tomlData(dic, k)
            tmpBouts = extractBouts(tracking, proto=dic["experiment"]["order"])
            a, b = boutTimeDist(inside=tmpBouts["inProd"], outside=tmpBouts["outProd"])
            c, d = boutLenDist(inside=tmpBouts["inProd"], outside=tmpBouts["outProd"])
            e, f = boutAngleDist(inside=tmpBouts["inProd"], outside=tmpBouts["outProd"])
            pref = pi(tracking, proto=dic["experiment"]["order"])
            data.append((dic["experiment"]["concentration"], [a,b,c,d,e,f, pref]))

    sortData = []
    for i in sorted(concentrations):
        buf = {"time": [], "len": [], "angle": []}
        prod = {"time": [], "len": [], "angle": []}
        pref = []
        for k, l in data:
            if i == k:
                prod["time"].extend(l[1])
                prod["len"].extend(l[3])
                prod["angle"].extend(l[5])
                buf["time"].extend(l[0])
                buf["len"].extend(l[2])
                buf["angle"].extend(l[4])
                pref.append(l[6])
        sortData.append((k, buf, prod, pref))

    ### Full plot
    fig, axs = plt.subplots(len(concentrations), 3, figsize=(10, 10), squeeze=False)
    for i, (j, k, l, __) in enumerate(sortData):
        sns.distplot(k["time"], label="Buffer", ax=axs[i, 0], norm_hist=True, kde=False, kde_kws={"gridsize": 1000, "clip": (0, 5)}, bins=np.linspace(0, 10, 100))
        sns.distplot(l["time"], label="Product", ax=axs[i, 0], norm_hist=True, kde=False, kde_kws={"gridsize": 1000, "clip": (0, 5)}, bins=np.linspace(0, 10, 100))
        axs[i, 0].set_xlim(0, 5)
        axs[i, 0].set_xlabel("Time (seconds)")
        axs[i, 0].legend()
        sns.distplot(k["len"], label="Buffer", ax=axs[i, 1], norm_hist=True, kde=False, kde_kws={"gridsize": 500}, bins=np.linspace(0, 50, 100))
        sns.distplot(l["len"], label="Product", ax=axs[i, 1], norm_hist=True, kde=False, kde_kws={"gridsize": 500}, bins=np.linspace(0, 50, 100))
        axs[i, 1].set_xlim(0, 25)
        axs[i, 1].set_xlabel("Length (mm)")
        axs[i, 1].legend()
        sns.distplot(k["angle"], label="Buffer", ax=axs[i, 2], norm_hist=True, kde=False, kde_kws={"gridsize": 500}, bins=np.linspace(-3, 3, 60))
        sns.distplot(l["angle"], label="Product", ax=axs[i, 2], norm_hist=True, kde=False, kde_kws={"gridsize": 500}, bins=np.linspace(-3, 3, 60))
        #axs[i, 2].set_xlim(-3.14, 3.14)
        axs[i, 2].set_xlabel("Angle(rad)")
        axs[i, 2].legend()
    fig.savefig(outPath + "/dist_plot.svg")          
                   
    ### Summary plot
    fig2, axs2 = plt.subplots(1, 3, figsize=(10, 5))
    control = [[],[],[]]
    for i, (j, k, l, __) in enumerate(sortData):
        sns.kdeplot(l["time"], label=str(j), ax=axs2[0])
        sns.kdeplot(l["len"], label=str(j), ax=axs2[1])
        sns.kdeplot(l["angle"], label=str(j), ax=axs2[2])

        control[0].extend(k["time"])
        control[1].extend(k["len"])
        control[2].extend(k["angle"])

    sns.kdeplot(control[0], label="control", ax=axs2[0], gridsize=1000, clip=(0,5))
    sns.kdeplot(control[1], label="control", ax=axs2[1], gridsize=500)
    sns.kdeplot(control[2], label="control", ax=axs2[2], gridsize=500)

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
    fig2.savefig(outPath + "/dist_sum.svg")          

    ### Preference index plot
    fig3, ax3 = plt.subplots()
    pindex = []
    concentrations = []
    for _, (i, __, __, j) in enumerate(sortData):
        concentrations.append(i)
        pindex.append(j)
    ax3.boxplot(pindex, positions=concentrations)
    ax3.set_xlabel("Concentration M")
    ax3.set_ylim(-1, 1)
    fig3.savefig(outPath + "/pref_index.svg")          
            

parser = argparse.ArgumentParser(description="Plot time, len, angle bouts distributions and preference index from toml files, sort by concentration uniquely")
parser.add_argument("path", nargs='+', help="Path to the toml files")
parser.add_argument("-o", default=None, dest="outPath", help="Path to save the figure")

args = parser.parse_args()
distribution(args.path, args.outPath)
