# np.memmap can't automatic adjust its size,
# here we implement our own Memmep.
# author: zhaoqianfeng
# e-mail: xxqianfeng@qq.com
import numpy as np
import os
import torch


class Memmap:
    r"""
    automatic capacity expansion memmap.
    If you create a Memmap with mode "w+" for write, you needn't declare it's shpae and dtype,
    Memmap will inference its shape and dtype the first time you call `add`.
    If you create a Memmap with mode "r" for read, you must give dtype and shape infomation on creation.

    Usage:
        # Create and Write a Memmap
        mmap = Memmap("/home/keys", mode="w+")
        a = torch.rand(10,64)
        mmap.add(a) 
        b = np.random.randn(38, 64)
        mmap.dump() # dump the file to disk

        # Read a Existed Memmap
        mmap = Memmap("/home/vals", mode="r", dtype=int, shape=(20000,))
    """

    def __init__(
        self,
        filename,
        mode = "r",
        dtype=None,
        shape=None,
    ):
        self.filename = filename
        self.mode = mode

        file_exists = os.path.exists(filename)
        if mode == "r" or mode == "r+":
            assert file_exists, "The memmap file %s dosen't exist" % filename
            assert dtype is not None, "must specify dtype when read a memmap"
            assert shape is not None, "must specify shape when read a memmap"
            if isinstance(shape, list):
                shape = tuple(shape)
            self.data = np.memmap(
                filename,
                dtype = self.convert_data_type(dtype),
                shape = shape,
                mode = mode,
            )
            self.size = shape[0]
            self.dtype = dtype
        else:
            self.data = None
            self.size = 0
            self.dtype = None


    @property
    def shape(self):
        r"""
        return the logical shape of a memmap.
        These function dont count redundant preallocated entries.

        for example, if we allocate [1000,5,8] space but the real entry size is 500,
        we will return [500, 5, 8] here.
        """
        return tuple([self.size] + list(self.data.shape[1:]))


    def add(self, data):
        # check the memmap read write mode
        assert self.mode == "r+" or self.mode == "w+", \
                "You can't write to a Memmap with {} mode.".format(self.mode)
        # allocate the Memmap file on the first time add function is called 
        if self.data is None:
            preallocated_shape = list(data.shape) if data.shape else [1]

            preallocated_shape[0] = 300000 # pre allocate [300000, ...] the first time
            preallocated_shape = tuple(preallocated_shape)
            self.dtype = self.convert_data_type(data.dtype)
            self.data = np.memmap(
                self.filename,
                dtype = self.dtype,
                shape = preallocated_shape,
                mode = self.mode,
            )
        
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        assert data.dtype == self.data.dtype, \
            "Inconsistent data types when add to memmap, require %s but add %s" % \
            (str(self.data.dtype), str(data.dtype)) 
            
        assert data.shape[1:] == self.data.shape[1:], \
            "Inconsistent data dimension when add to memmap, require %s but add %s" % \
            (str(self.data.shape[1:]), str(data.shape[1:]))

        data_shape = data.shape if data.shape else (1,)
        need_resize = False
        now_capacity = self.data.shape[0]
        new_capacity = now_capacity
        while data_shape[0] + self.size >= new_capacity:
            need_resize = True
            # Take a more aggressive pre allocation strategy when there are fewer entries
            if new_capacity < 5000000: 
                new_capacity = 2 * new_capacity
            else:
            # Take a more conservative pre allocation strategy when there are many entries
                new_capacity = int(new_capacity * 1.5)
        
        if need_resize:
            new_shape = [new_capacity] + list(self.data.shape[1:])
            new_shape = tuple(new_shape)
            new_memory_footprint = self.data.dtype.itemsize
            for x in new_shape:
                new_memory_footprint *= x
            
            self.data.base.resize(new_memory_footprint)
            self.data.flush()
            self.data = np.memmap(
                self.filename,
                dtype = self.dtype,
                mode = "r+",
                shape = new_shape,
        )

        self.data[self.size:self.size+data_shape[0]] = data
        self.size += data_shape[0]


    def drop_redundant(self):
        r"""
        trim the memmap, discard redundant preallocated entries
        """
        if self.size != self.data.shape[0]:
            new_shape = self.shape
            new_memory_footprint = self.data.dtype.itemsize
            for x in new_shape:
                new_memory_footprint *= x 
        
            self.data.base.resize(new_memory_footprint)
            self.data.flush()
            self.data = np.memmap(
                self.filename,
                dtype = self.dtype,
                mode = "r+",
                shape = new_shape,
        )
        

    def dump(self):
        r""" 
        when we dump the Memmap to disk, we dicard redundant preallocated entries.
        It means we trim the memmap to `self.size` entries
        """
        self.drop_redundant()

        
    @staticmethod
    def convert_data_type(data_type):
        r""" convert an input data dtype to numpy compatible dtype """
        data_type_convert_dict = {
            np.float32: np.float32,
            np.float16: np.float16,
            np.dtype('float32'): np.float32,
            np.dtype('float16'): np.float16,
            torch.float: np.float32,
            torch.float32: np.float32,
            torch.float16: np.float16,
            "float32": np.float32,
            "float16": np.float16,
            # string to dtype is needed when restore dtype from json file
            str(np.float32): np.float32,
            str(np.float16): np.float16,
            str(torch.float): np.float32,
            str(torch.float32): np.float32,
            str(torch.float16): np.float16,
            np.int16: int,
            np.int32: int,
            np.int64: int,
            np.dtype('int64'):int,
            np.dtype('int32'): int,
            np.dtype('int16'): int,
            np.int_: int,
            int: int,
            torch.int64: int,
            torch.int32: int,
            torch.int: int,
            str(np.int16): int,
            str(np.int32): int,
            str(np.int64): int,
            str(np.int_): int,
            "<class 'int'>": int,
            str(torch.int64): int,
            str(torch.int32): int,
            str(torch.int): int,
            "int": int,
        }

        assert data_type in data_type_convert_dict, \
                "Unsupported data type when convert dtype for memmap!" 
        return data_type_convert_dict[data_type] 


        
        
