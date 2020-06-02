import toml
import numpy as np



info = { "title" : "test", "path" : "path", "author" : "Benjamin Gallois" }
fish = { "age" : 20, "date" : "",  "type" : "wt", "remark" : ""}
experiment = {"date" : "", "product" : "", "concentration" : 10, "interface" : 510, "order" : "BRBL", "buffer1" : [0, 9999], "product1" : [10000, 19999], "buffer2" : [20000,29999], "product2" : [30000, 39999]}

metadata = { "image" : range(40000), "time" : np.linspace(10, 11, 40000).tolist() }


position = (np.random.rand(40000)*600).tolist()
#position = [0]*20000 + [900]*20000
position2 = [j for i, j in enumerate(position) if i%2 ==0]
trackingData = { "xHead" : position, "yHead" : position, "imageNumber" : range(40000) }
trackingData2 = { "xHead" : position2, "yHead" : position2, "imageNumber" : range(0, 40000, 2) }
tracking = {"Fish_0" : trackingData, "Fish_1" : trackingData2}


dic = {"info" : info, "fish" : fish, "experiment" : experiment, "metadata" : metadata, "tracking" : tracking}

with open("testToml.toml", "w") as f:
    toml.dump(dic, f)
