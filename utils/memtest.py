from memmap import Memmap
import numpy as np

mmap = Memmap("/data1/zhaoqf/tests/mmap", dim=1024, dtype="float32", mode="w+")
array = np.ones((1,1024))
mmap.append(array)
array = np.ones((1,1024)) * 2
mmap.append(array)
array = np.ones((1,1024)) * 3
mmap.append(array)
print(mmap.end)
print(mmap.capacity)
mmap.dump()
print(mmap.data)