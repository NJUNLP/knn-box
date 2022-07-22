from datastore.datastore import Datastore
from utils.utils import global_datastores
from test.add_value import model
import numpy as np


if "my_ds" not in global_datastores:
    ds = Datastore("/data1/zhaoqf/ds_path")
    global_datastores["my_ds"] = ds
else:
    ds = global_datastores["my_ds"]



if __name__ == "__main__":
    keys = np.ones((300000,768))
    ds.add_key(keys)
    ml = model()
    ml.forward(keys.shape)
    keys = np.ones((20000,768)) * 8 
    ml.forward(keys.shape)
    ds.add_key(keys)
    ds.dump()
    print(ds.values.data)
    print(ds.keys.data)
