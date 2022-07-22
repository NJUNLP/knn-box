import numpy as np
from utils.utils import global_datastores
from datastore.datastore import Datastore


if "my_ds" not in global_datastores:
    ds = Datastore("/data1/zhaoqf/ds_path")
    global_datastores["my_ds"] = ds
else:
    ds = global_datastores["my_ds"]


class model:
    def forward(self, key_size):
        values = key_size[0]*np.ones((key_size[0],1))
        ds.add_value(values)