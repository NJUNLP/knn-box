# np.memmap can't automatic adjust its size,
# here we implement our own Memmep.
import numpy as np
import torch

class Memmap:
    r"""
    automatic capacity expansion memmap.
    Usage:
        mmap = Memmap("/home/keys", dtype="float32")
        t = np.Array(3)
        mmap.append(t)
    """

    def __init__(
        self,
        filename,
        dim,
        dtype,
        capacity = 65536, 
        mode = "r+",
    ):
        self.end = 0 
        self.capacity = capacity
        self.dim = dim
        self.filename = filename
        self.dtype = dtype
        self.mode = mode
        self.data = np.memmap(
            filename,
            dtype = dtype,
            mode = mode,
            shape = (self.capacity, self.dim)
        )
    

    def add(self, data):
        data_shape = data.shape
        need_resize = False
        while data_shape[0] + self.end >= self.capacity:
            need_resize = True
            self.capacity *= 2
        if need_resize:
            self.data.base.resize(
                self.capacity * self.dim * self.data.dtype.itemsize
            )
            self.data.flush()
            self.data = np.memmap(
                self.filename,
                dtype = self.dtype,
                mode = "r+",
                shape = (self.capacity, self.dim)
        )
        
        data = data.detach().cpu().numpy()
        self.data[self.end:self.end+data_shape[0]] = data
        # self.data.flush()
        self.end += data_shape[0]


    def dump(self):
        print("saving...")
        self.trim()
        print("save down.")
        


    def trim(self):
        if self.end != self.capacity:
            self.capacity = self.end
            self.data.base.resize(
                self.capacity * self.dim * self.data.dtype.itemsize
            )
            self.data.flush()
            self.data = np.memmap(
                self.filename,
                dtype = self.dtype,
                mode = "r+",
                shape = (self.capacity, self.dim)
        )
        




        
        
