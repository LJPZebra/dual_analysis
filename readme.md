# Dual analysis CLI and workflow

Dual command line interface is made to standardize (across version and user) the analysis of experiment perform on Dual.

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

### Extract timestamps (old dual version)

In the old version of the Dual control software, the Timestamps.txt was not generated but embedded in the frame. To extract the metadata and create the Timestamps.txt used the CLI command:
```
python extractTimestamp.py path_to_the_raw_folder 
```

### Generate the toml file

Use the cli command:
```
python createToml.py path_to_the_raw_folder --name outputName -o dest
```

> For each experiment, for product start image is manually set to account for a possible delay due to air in the syringe or variable transitory phase.

### Loop on several folders

Use the cli command:
```
python3 createToml.py "$(python3 listPath.py root)"
```

