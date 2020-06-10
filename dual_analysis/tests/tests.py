import pytest
import random
import sys
import numpy as np
import pandas as pd
sys.path.append('.')
from distribution import *

def test_mod1():
    test = []
    test.append(Mod1(np.pi, 0.5*np.pi))
    test.append(Mod1(0.5*np.pi, np.pi))
    test.append(Mod1(7*0.25*np.pi, 0.25*np.pi))
    test.append(Mod1(0.25*np.pi, 7*0.25*np.pi))
    np.testing.assert_equal(np.around(test, 4), np.around([-0.5*np.pi, 0.5*np.pi, 0.5*np.pi, -0.5*np.pi], 4))

def test_mod():
    test = [7*0.25*np.pi, 0.25*np.pi, 0.5*np.pi]
    np.testing.assert_equal(np.around(Mod(test), 4), np.around([-0.5*np.pi, -0.25*np.pi], 4))

def test_tomlData():
    tracking = pandas.read_csv("tests/Test_Run/Tracking_Result/tracking.txt", sep='\t')
    for i in range(np.max(tracking.id.values) + 1):
        test = tomlData("tests/test.toml", i).dropna(inplace=True)
        ref = concatenateData("tests/Test_Run/", i).dropna(inplace=True)
        assert test == ref
        dic = toml.load("tests/test.toml")
        test = tomlData(dic, i).dropna(inplace=True)
        ref = concatenateData("tests/Test_Run/", i).dropna(inplace=True)
        assert test == ref
    
def test_concatenateData():
    tracking = pandas.read_csv("tests/Test_Run/Tracking_Result/tracking.txt", sep='\t')
    distance = pandas.read_csv("tests/Test_Run/distance.txt", sep='\t')
    test = []
    ref = []
    for iD in range(np.max(tracking.id.values) + 1):
        tmp = concatenateData("tests/Test_Run/", iD)
        for i in range(50):
            index = random.randint(0, len(tmp.imageNumber.values))
            tmpTest = tmp[tmp.imageNumber == index]
            if tmpTest.empty:
               continue
            else:
                test.append(tmpTest.distance.values[0])
            tmpDist = distance[(distance.imageNumber == index) & (distance.id == iD)]
            if tmpTest.cycle.values[0] == -1:
                ref.append(0)
            elif tmpDist.empty:
               ref.append(np.nan)
            elif tmpTest.cycle.values[0] == 1:
                ref.append(tmpDist.distance.values[0]-0.001)
            elif tmpTest.cycle.values[0] == 2:
                ref.append(tmpDist.distance.values[0]+0.001)
            else:
                ref.append(tmpDist.distance.values[0])
    np.testing.assert_equal(ref, test)
    
def test_crossing():
    cycle = 10*[-1] + 10*[1] 
    xHead = np.sin(np.linspace(0, 10, 20))*500 + 500
    distance = 10*[0] + list(np.sin(np.linspace(0, 10, 10))*500)
    ref = pd.DataFrame({"xHead": xHead, "distance": distance, "cycle": cycle})
    refCross = [1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1.,0., 0., 1.]
    test = findCrossing(ref)
    np.testing.assert_equal(refCross, test)
    
def test_pi_cycle():
    cycle = 10*[1] + 10*[2] 
    time = np.linspace(0, 10, 20)
    distance = list(np.sin(np.linspace(0, 10, 20))*500)
    xHead = np.sin(np.linspace(0, 10, 20))*500 + 500
    ref = pd.DataFrame({"time": time, "distance": distance, "cycle": cycle, "xHead": xHead})
    assert pi(ref, proto="BLBR") == - pi(ref, proto="BRBL")
    
def test_pi_0():
    cycle = 10*[1] + 10*[2] 
    time = np.linspace(0, 19, 20)
    distance = np.linspace(-500, -1, 20)
    xHead = np.sin(np.linspace(0, 10, 20))*500 + 500
    ref = pd.DataFrame({"time": time, "distance": distance, "cycle": cycle, "xHead": xHead})
    assert pi(ref) == 0 
    
def test_pi_1_BLBR():
    cycle = 10*[1] + 10*[2] 
    time = np.linspace(0, 19, 20)
    distance = list(np.linspace(-500, -1, 10)) + list(np.linspace(1, 500, 10))
    xHead = np.sin(np.linspace(0, 10, 20))*500 + 500
    ref = pd.DataFrame({"time": time, "distance": distance, "cycle": cycle, "xHead": xHead})
    assert pi(ref) == 1
    
def test_pi_1_BRBL():
    cycle = 10*[1] + 10*[2] 
    time = np.linspace(0, 19, 20)
    distance = list(np.linspace(-500, -1, 10)) + list(np.linspace(1, 500, 10))
    xHead = np.sin(np.linspace(0, 10, 20))*500 + 500
    ref = pd.DataFrame({"time": time, "distance": distance, "cycle": cycle, "xHead": xHead})
    assert pi(ref, proto="BRBL") == -1 

def test_pi_negative():
    cycle = 10*[1] + 10*[2] 
    time = np.linspace(3791190449921996, 5091192453224933, 20)
    distance = list(np.linspace(-500, -1, 10)) + list(np.linspace(1, 500, 10))
    xHead = np.sin(np.linspace(0, 10, 20))*500 + 500
    ref = pd.DataFrame({"time": time, "distance": distance, "cycle": cycle, "xHead": xHead})
    assert np.isnan(pi(ref, proto="BRBL")) 
