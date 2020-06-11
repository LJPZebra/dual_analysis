import numpy as np
from collections import namedtuple
import seaborn as sns
import glob
import numpy as np
import pandas
import matplotlib.pyplot as plt
from matplotlib import rc
import scipy
from scipy.signal import find_peaks
from scipy.signal import peak_widths
import matplotlib as mpl
from matplotlib.patches import Rectangle
import toml


def Mod1(a, b):
    '''
    Compute the minimal difference between to angles.
    
    Parameters
    ----------
    a : float
        First angle 0 to 2pi.
    b : float
        Second angle 0 to 2pi.

    Returns
    -------
    TYPE
        (a - b) minimal angle 0 to pi.

    '''
    def modul(a): return a - 2 * np.pi * np.floor(a / (2 * np.pi))
    a = modul(a)
    b = modul(b)
    return -(modul(a - b + np.pi) - np.pi)


def Mod(a):
    '''
    Compute the minimal difference for an array of angles.
    
    Parameters
    ----------
    a : list
        List of angles 0 to 2pi.

    Returns
    -------
    Array
        Array of minimal difference angles 0 to pi.

    '''
    diff = np.zeros(len(a)-1)
    for i, j in enumerate(a):
        if i < len(diff):
            diff[i] = Mod1(a[i+1], a[i])
    return diff


def findCrossing(dataframe):
    '''
    Find the moment where the fish cross the interface from the distance to the
    interface and the head position.

    Parameters
    ----------
    dataframe : pandas dataframe
        Tracking datafame.

    Returns
    -------
    cross : list
        List where values 1 are corresponding to a crossing.

    '''
    dataframe.dropna(inplace=True)
    cross = np.zeros(len(dataframe.xHead.values))
    tmp = dataframe.distance.values
    for i, j in enumerate(tmp):
        if np.sign(tmp[i-1]) != np.sign(tmp[i]):
            cross[i] = 1
        if dataframe["cycle"].values[i] == -1:
            if (dataframe.xHead.values[i-1] <= 500 and dataframe.xHead.values[i] > 500 ) or (dataframe.xHead.values[i-1] >= 500 and dataframe.xHead.values[i] < 500 ):
                cross[i] = 1
    cross[0] = 1
    cross[-1] = 1
    return cross


def tomlData(path, iD=0):
    '''
    Open a toml file of concatenate data from a Dual experiment

    Parameters
    ----------
    path : str
        path to a toml file, or dict.
    iD : int, optional
        Id of the object in to extract from the data. The default is 0.

    Returns
    -------
    tracking : pandas dataframe
        Dataframe containing the data for the selected object.

    '''
    # Import data
    factorDist = 30/440
    factorTime = 1e-9
    if not isinstance(path, dict):
        dicTracking = toml.load(path)
    else:
        dicTracking = path
    tracking = pandas.DataFrame(dicTracking["tracking"]["Fish_" + str(iD)])
    time = pandas.DataFrame(dicTracking["metadata"])
    tracking["time"]  = time["time"].iloc[tracking["imageNumber"].values].values

    # Separate between cycles:
    # -1 control buffer
    # 1 first cycle
    # 2 second cycle
    bufferCycle = dicTracking["experiment"]["buffer1"]
    firstCycle = dicTracking["experiment"]["product1"]
    secondCycle = dicTracking["experiment"]["product2"]
    tracking["cycle"] = np.zeros(len(tracking.imageNumber.values))
    for i, j in enumerate(tracking.imageNumber.values):
        if j > bufferCycle[0] and j < bufferCycle[1]:
            tracking["cycle"].iloc[i] = -1
            tracking["distance"].iloc[i] = 0
        elif j > firstCycle[0] and j < firstCycle[1]:
            tracking["cycle"].iloc[i] = 1
            tracking["distance"].iloc[i] -= 0.001
        elif j > secondCycle[0] and j < secondCycle[1]:
            tracking["cycle"].iloc[i] = 2
            tracking["distance"].iloc[i] += 0.001
    
    cross = findCrossing(tracking)
    tracking["cross"] = cross

    return tracking

def concatenateData(path, iD=0):
    '''
    Concatenate all the data from a Dual experiment in one dataframe.

    Parameters
    ----------
    path : str
        path to a Dual analysis folder.
    iD : int, optional
        Id of the object in to extract from the data. The default is 0.

    Returns
    -------
    tracking : pandas dataframe
        Dataframe containing the data for the selected object.

    '''
    # Import data
    time = pandas.read_csv(path + "Timestamps.txt", sep='\t', header=None)
    meta = pandas.read_csv(path + "/Milestones.txt", sep='\t', header=None)
    images = glob.glob(path + "/Frame*pgm")
    tracking = pandas.read_csv(path + "/Tracking_Result/tracking.txt", sep='\t')
    tracking = tracking[tracking.id == iD]
    dist = pandas.read_csv(path + "distance.txt", sep='\t', header=0)
    dist = dist[dist.id == iD]
    factorDist = 30/440
    factorTime = 1e-9
    tracking["time"]  = time[1].iloc[tracking["imageNumber"].values].values

    # Clean the distance between fish and interface
    for i, j in enumerate(dist.distance.values):
        '''if i > 0 and i < len(dist) - 1:
            if j == 0 and np.sign(dist.distance.values[i - 1]) != 0 and ( np.sign(dist.distance.values[i - 1]) == np.sign(dist.distance.values[i + 1]) ):
                dist.loc[i, "distance"] = np.nan'''
    tmp = []
    for i, j in enumerate(tracking.imageNumber.values):
        tmpDist = dist[dist.imageNumber == j].distance
        if not tmpDist.empty:
            tmp.append(tmpDist.values[0])
        else:
            tmp.append(np.nan)
    tracking["distance"] = tmp
    #tracking["distance"].interpolate(inplace=True)
    meanInterface = np.median(tracking[tracking.distance == 0].xHead.values)
    #tracking.iloc[tracking[ (tracking.distance == 0) & ((tracking.xHead > (meanInterface + 100)) | (tracking.xHead < (meanInterface - 100))) ].index].distance = np.nan

    # Separate between cycles:
    # -1 control buffer
    # 1 first cycle
    # 2 second cycle
    bufferCycle = (meta[0][1], meta[0][2])
    firstCycle = (meta[0][3] + 2000, meta[0][4])
    secondCycle = (meta[0][7] + 2000, 50000)
    tracking["cycle"] = np.zeros(len(tracking.imageNumber.values))
    for i, j in enumerate(tracking.imageNumber.values):
        if j > bufferCycle[0] and j < bufferCycle[1]:
            tracking["cycle"].iloc[i] = -1
            tracking["distance"].iloc[i] = 0
        elif j > firstCycle[0] and j < firstCycle[1]:
            tracking["cycle"].iloc[i] = 1
            tracking["distance"].iloc[i] -= 0.001
        elif j > secondCycle[0] and j < secondCycle[1]:
            tracking["cycle"].iloc[i] = 2
            tracking["distance"].iloc[i] += 0.001
    
    cross = findCrossing(tracking)
    tracking["cross"] = cross

    return tracking


def extractTraj(dataframe, addCleaning=True):
    '''
    This function extracts the trajectories from one crossing of the interface to the next crossing. In the case of buffer | buffer the virtual crossing occurs at x = 500.
    It outputs a dictionnary with each case. Each case contains a list of dataframe for each trajectory.
    Additionals cleaning rules can be added.
    
    Parameters
    ----------
    dataframe : pandas dataframe
        Concatenated dataframe.
    addCleaning : bool, optional
        Clean the data. The default is True.

    Returns
    -------
    None.
    
    '''
    def clean(dataframe):
        '''
        Cleaning rules to ignore interface extraction artefact:
            * Exclude crossings that are not near the center
        '''
        for i, j in enumerate(dataframe.cross.values):
            if (j == 1) and ((dataframe.xHead.values[i] < 300) or (dataframe.xHead.values[i] > 700)):
                dataframe["cross"].values[i] = 0
                
    # Select cycle
    if addCleaning: clean(dataframe)
    cycle1 = dataframe[dataframe.cycle == 1]
    cycle1["cross"].values[0] = 1
    cycle1["cross"].values[-1] = 1
    cycle2 = dataframe[dataframe.cycle == 2]
    cycle2["cross"].values[0] = 1
    cycle2["cross"].values[-1] = 1
    cycleC = dataframe[dataframe.cycle == -1]
    cycleC["cross"].values[0] = 1
    cycleC["cross"].values[-1] = 1
    
    insideProduct = []
    outsideProduct = []
    insideControl = []
    outsideControl = []
    
    prev = 0
    for i, j in enumerate(cycle1.cross.values):
        if i == 0: continue
        if j == 1:
            if cycle1.distance.values[prev] <= 0:
                insideProduct.append(cycle1.iloc[prev : i+1])
            elif cycle1.distance.values[prev] > 0:
                outsideProduct.append(cycle1.iloc[prev : i+1])
            prev = i
        
    prev = 0
    for i, j in enumerate(cycle2.cross.values):
        if i == 0: continue
        if j == 1:
            if cycle2.distance.values[prev] >= 0:
                insideProduct.append(cycle2.iloc[prev : i+1])
            elif cycle2.distance.values[prev] < 0:
                outsideProduct.append(cycle2.iloc[prev : i+1])
            prev = i
            
    prev = 0
    for i, j in enumerate(cycleC.cross.values):
        if i == 0: continue
        if j == 1:
            if cycleC.xHead.values[prev] >= 500:
                insideControl.append(cycleC.iloc[prev : i+1])
            elif cycleC.xHead.values[prev] < 500:
                outsideControl.append(cycleC.iloc[prev : i+1])
            prev = i
    
    return {"inProd": insideProduct, "outProd": outsideProduct, "inCont": insideControl, "outCont": outsideControl}


def pi(dataframe, proto="BLBR"):
    '''
    Compute the preference index.

    Parameters
    ----------
    dataframe : pandas datafame.
        Concatenate dataframe.
    proto : str, optional
        Experiment cycles order. The default is "BLBR".

    Returns
    -------
    Int
        Preference index -1:1.

    '''
    prod = 0
    buff = 0
    cycle1 = dataframe[dataframe.cycle == 1]
    for t, c in zip(np.diff(cycle1.time.values), cycle1.distance.values):
            if t > 50*10E7:
                continue
            if c <= 0:
                prod += t 
            elif c > 0:
                buff += t 
    cycle2 = dataframe[dataframe.cycle == 2]
    for t, c in zip(np.diff(cycle2.time.values), cycle2.distance.values):
            if t > 50*10E7:
                continue
            if c >= 0:
                prod += t 
            elif c < 0:
                buff += t
    if prod + buff == 0:
        return np.nan
    if proto == "BRBL":
        return -(prod-buff) / (buff+prod)
    elif proto == "BLBR":
        return (prod-buff) / (buff+prod)
    else:
        return np.nan


def extractBouts(dataframe, proto="BLBR"):
    '''
    Extract fish bouts.

    Parameters
    ----------
    dataframe : pandas dataframe
        Concatenate dataframe.
    proto : str, optional
        Experiment cycles order. The default is "BLBR".

    Returns
    -------
    Dict
        Dictionary containing dataframe of bout sequences.

    '''
    def bouts(dataframe):
        '''
        Find peaks on the velocity signal

        Parameters
        ----------
        dataframe : pandas dataframe
            Concatenate dataframe.

        Returns
        -------
        peaks : array
            Index of the peaks.

        '''
        l = ((30/440)*(np.diff(dataframe.xHead.values)**2 + np.diff(dataframe.yHead.values)**2)**0.5) / (1e-9*np.diff(dataframe["time"]))
        peaks, __ = find_peaks(l, height=8, prominence=8, distance=5)
        return peaks
    
    def cleanRules(dataframes):
        '''
        Bouts additional cleaning rules.

        Parameters
        ----------
        dataframes : pandas dataframe
            Concatenated dataframe.

        Returns
        -------
        dataframes : pandas dataframe
            Cleaned dataframe.

        '''
        remove = []
        for i, j in enumerate(dataframes):
            if np.max(np.diff(j.imageNumber.values)) >= 5:
                remove.append(i)
                if i > 0: remove.append(i - 1)
                if i < len(j.time.values)-1: remove.append(i + 1)
        remove = list(set(remove))
        for i in sorted(remove, reverse=True):
            #del dataframes[i]
            pass
        return dataframes
                
    # Select cycle
    cycle1 = dataframe[dataframe.cycle == 1]
    cycle2 = dataframe[dataframe.cycle == 2]
    cycleC = dataframe[dataframe.cycle == -1]
    '''cycle1["cross"].values[0] = 1
    cycle1["cross"].values[-1] = 1
    cycle2["cross"].values[0] = 1
    cycle2["cross"].values[-1] = 1
    cycleC["cross"].values[0] = 1
    cycleC["cross"].values[-1] = 1'''
    
    insideProduct = []
    outsideProduct = []
    insideControl = []
    outsideControl = []
    
    cycle1Peaks = bouts(cycle1)
    tmp = []
    for i, j in enumerate(cycle1Peaks):
        if j == len(cycle1.imageNumber.values)-1 or i == len(cycle1Peaks)-1: break
        tmp.append(cycle1.iloc[j+1 : cycle1Peaks[i+1]])
    tmp = cleanRules(tmp)
        
    for i, j in enumerate(tmp):
        if j.distance.values[0] <= 0:
            insideProduct.append(j)
        elif j.distance.values[0] > 0:
            outsideProduct.append(j)
        
    cycle2Peaks = bouts(cycle2)
    tmp = []
    for i, j in enumerate(cycle2Peaks):
        if j == len(cycle2.imageNumber.values)-1 or i == len(cycle2Peaks)-1: break
        tmp.append(cycle2.iloc[j+1 : cycle2Peaks[i+1]])
    tmp = cleanRules(tmp)
        
    for i, j in enumerate(tmp):
        if j.distance.values[0] >= 0:
            insideProduct.append(j)
        elif j.distance.values[0] < 0:
            outsideProduct.append(j)
            
    cycleCPeaks = bouts(cycleC)
    tmp = []
    for i, j in enumerate(cycleCPeaks):
        if j == len(cycleC.imageNumber.values)-1 or i == len(cycleCPeaks)-1: break
        tmp.append(cycleC.iloc[j+1 : cycleCPeaks[i+1]])
    tmp = cleanRules(tmp)
   
    for i, j in enumerate(tmp):
        if j.xHead.values[0] >= 500:
            insideControl.append(j)
        elif j.xHead.values[0] < 500:
            outsideControl.append(j)
    
    return {"inProd": insideProduct, "outProd": outsideProduct, "inCont": insideControl, "outCont": outsideControl}


def extractBoutsSimple(dataframe, proto="BLBR"):
    
    def bouts(dataframe):
        l = ((30/440)*(np.diff(dataframe.xHead.values)**2 + np.diff(dataframe.yHead.values)**2)**0.5) / (1e-9*np.diff(dataframe["time"]))
        #peaks, __ = find_peaks(-l, prominence=5, distance=3)
        peaks, __ = find_peaks(l, height=8, prominence=8, distance=5)
        return peaks
    def cleanRules(dataframes):
        remove = []
        for i, j in enumerate(dataframes):
            if np.max(np.diff(j.imageNumber.values)) >= 5:
                remove.append(i)
                if i > 0: remove.append(i - 1)
                if i < len(j.time.values)-1: remove.append(i + 1)
        remove = list(set(remove))
        for i in sorted(remove, reverse=True):
            print(dataframes[i])
            del dataframes[i]
        return dataframes
                
    cyclePeaks = bouts(dataframe)
    tmp = []
    for i, j in enumerate(cyclePeaks):
        if j == len(dataframe.imageNumber.values) - 1 or i == len(cyclePeaks)-1: break
        tmp.append(dataframe.iloc[j+1 : cyclePeaks[i+1]])
    #tmp = cleanRules(tmp)
        
    return tmp


def boutTimeDist(inside, outside):
    '''
    Extract time inter bouts.

    Parameters
    ----------
    inside : list
        List of bout dataframes.
    outside : list
        List of bout dataframes.

    Returns
    -------
    tBuff : list
        List of time inter bouts.
    tProd : list
        List of time inter bouts.

    '''
    tBuff = []
    for i, j in enumerate(outside):
        tBuff.append((j.time.values[-1] - j.time.values[0])*1e-9)
        
    tProd = []
    for i, j in enumerate(inside):
        tProd.append((j.time.values[-1] - j.time.values[0])*1e-9)
    return tBuff, tProd

def boutAngleDist(inside, outside):
    '''
    Extract reorientation angle between bouts.

    Parameters
    ----------
    inside : list
        List of bout dataframes.
    outside : list
        List of bout dataframes.

    Returns
    -------
    tBuff : list
        List of reorientation angles.
    tProd : list
        List of reorientation angles.

    '''
    rejected = 0
    tBuff = []
    for i, j in enumerate(outside):
        #tBuff.append(np.sum(Mod(j.tHead.values)))
        #if j.xHead.values[0] > 800 or j.xHead.values[0] < 200 or j.yHead.values[0] > 400 or j.yHead.values[0] < 100:
        #    pass
        if abs(np.sum(Mod(j.tHead.values))) > np.pi: #- Mod1(j.tHead.values[-1], j.tHead.values[0]) > np.pi/25:
            rejected += 1
            tBuff.append(np.sum(Mod1(j.tHead.values[-1],j.tHead.values[-1])))
        else:
            tBuff.append(np.sum(Mod(j.tHead.values)))
        
    tProd = []
    for i, j in enumerate(inside):
        #tProd.append(np.sum(Mod(j.tHead.values)))
        #if j.xHead.values[0] > 800 or j.xHead.values[0] < 200 or j.yHead.values[0] > 400 or j.yHead.values[0] < 100:
         #   pass
        if abs(np.sum(Mod(j.tHead.values))) > np.pi: #- Mod1(j.tHead.values[-1], j.tHead.values[0]) > np.pi/25:
            rejected += 1
            tProd.append(np.sum(Mod1(j.tHead.values[-1],j.tHead.values[-1])))
        else:
            tProd.append(np.sum(Mod(j.tHead.values)))
                     
    #print("Rejected", rejected/(len(tBuff) + len(tProd)))
    return tBuff, tProd

def boutLenDist(inside, outside, proto="BLBR"):
    '''
    Extract length inter bouts.

    Parameters
    ----------
    inside : list
        List of bout dataframes.
    outside : list
        List of bout dataframes.

    Returns
    -------
    tBuff : list
        List of length inter bouts.
    tProd : list
        List of length inter bouts.

    '''
        
    tbuff = []
    for i, j in enumerate(outside):
        tbuff.append((30/440)*np.sum((np.diff(j.xHead.values)**2 + np.diff(j.yHead.values)**2)**0.5))
        
    tprod = []
    for i, j in enumerate(inside):
        tprod.append((30/440)*np.sum((np.diff(j.xHead.values)**2 + np.diff(j.yHead.values)**2)**0.5))
                     
    return tbuff, tprod
