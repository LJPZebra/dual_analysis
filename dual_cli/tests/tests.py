import pytest
import sys
import numpy as np
import pandas as pd
sys.path.append('.')
import subprocess
import toml
    
def test_add_distance():
    tracking = pd.read_csv("tests/2020-02-20/Test_Run/Tracking_Result/tracking.txt", sep='\t')
    distance = pd.read_csv("tests/2020-02-20/Test_Run/distance.txt", sep='\t')

    process = subprocess.call(["../venv/bin/python3", "createToml.py", "tests/2020-02-20/Test_Run/", "-o", "tests/"])
    f = toml.load("tests/20200220Test_Run.toml")

    for l in range(np.max(tracking.id.values + 1)):
        for k, (i, j) in enumerate(zip(f["tracking"]["Fish_" + str(l)]["imageNumber"], f["tracking"]["Fish_" + str(l)]["distance"])):
            if not np.isnan(j):
                assert np.around((j), 3) == np.around(distance[(distance.imageNumber == i) & (distance.id == l)].distance.values[0], 3)
            assert np.around(f["tracking"]["Fish_" + str(l)]["xHead"][k], 3) == np.around(tracking[(tracking.imageNumber == i) & (tracking.id == l)].xHead.values[0], 3)


def test_info():
    f = toml.load("tests/20200220Test_Run.toml")

    assert f["experiment"]["product"] == "citricacide"
    assert f["experiment"]["concentration"] == 0.02
    assert f["experiment"]["order"] == "BLBR"
    assert f["fish"]["age"] == 7
