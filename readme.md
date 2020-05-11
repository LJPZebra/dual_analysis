# Dual analysis CLI

Dual command line interface is made to standardize (across version and user) the analysis of experiments perform on Dual.

## Folder architecture description

### Dual control software output

The software control output a folder with:
* `Frame_xxxxxx.pgm` the images with some metadata include at the last line
* `Milestones.txt` the frame number and timestamp for each cycle
* `Protocol.txt` the protocol used
* `Timestamps` (from version 2.01, see how to generate timestamp below) frame number and corresponding timestamp.

### Fast Track output

Experiments are analyzed with Fast Track that generates a Tracking_Result folder with the data in a tracking.txt file.
> Experiment are tracked with Fast Track and manually corrected inside the software if necessary

## Create the toml file from raw folder

To standardize the result of each experiment, all the information are parsed and written in a toml file as follow:
```
[info]
title = "../test/"
author = "Benjamin Gallois"
[fish]
age = 6
date = ""
type = "wt"
remark = ""
[experiment]
date = ""
product = "atp"
concentration = 125.0
position = ""
order = ""
buffer1 = [ 1031, 8712,]
buffer2 = [ 18468, 26166,]
product1 = [ 9744, 17432,]
product2 = [ 27200, 34885,]
[metadata]
image = [ ]
time = [ ]
[tracking.Fish_number]
*/All the tracking result/*
```

### Activate the virtual environment
```
source bin/activate
```

### Extract timestamps (old dual version)

In the old version of the Dual control software, the Timestamps.txt was not generated but embedded in the frame. To extract the metadata and create the Timestamps.txt used the CLI command:
```
python3 extractTimestamp.py path_to_the_raw_folder 
```

### Generate the toml file

Use the cli command:
```
python3 createToml.py path_to_the_raw_folder --name outputName -o dest
```

> For each experiment, for product start image is manually set to account for a possible delay due to air in the syringe or variable transitory phase.

### Update the toml file with new tracking data
Use the cli command:
```
python3 updateToml.py path_to_the_raw_folder -o dest
```


### Loop on several folders

Use the cli command:
```
python3 listPath.py root | xargs python3 createToml.py -o dest --erase True
```
And,
```
python3 listPath.py root | xargs python3 extractTimestamp.py
```


## Analysis worflow

The analysis workflow can change for the version of Dual used to performed the experiments.

* [auto] Extract the timestamps
* [auto] Perform the tracking with Fast Track
* [man] Check the tracking
* [auto] Generate the toml file
* [man] Adjust manually the beginning of the product cycle, the order, the interface position and check the toml file
* [auto] Preference index analysis
* [auto] Archive the experiment in the git folder

> The archive repository as the following convention for the commit:
> [add] for the addition of new experiments, [cor] for the correction in archived toml files, [upt] for an update on all the toml files. 

## Analysis tools

If you want to extract only the toml file that correspond to some criterions
```
python3 list.py files.toml --product products --concentration concentrations --age min max
```
And for  the analysis:
```
python3 list.py files.toml --product products --concentration concentrations --age | xargs python analysis.py
```

If you want to plot the trajectory for one or several fish
```
python3 list.py files.toml --product products --concentration concentrations --age | xargs python trace.py
```

## Analysis

### Preference Index

The preference index is defined as follow:

$$ pi = \frac{t_{product} - t_{buffer}}{t_{product} + t_{buffer}}$$

And,
$$ pi = \frac{t_{left/right} - t_{rifgt/left}}{t_{left/right} + t_{right/left}}$$

for the buffer cycle.
The preference index is an indicator of the preference of the fish.

### Activity

The activity is defined as follow:

$$ a = <\Delta l> _{cycle} $$



