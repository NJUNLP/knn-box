from libds.utils.cache import TensorCache
import torch
import time

t = torch.rand(1000000,512)
distance = torch.rand(1000000,2)
indices = torch.rand(1000000,2)
values = {"distance": distance, "indices":indices}
print(values)
cache = TensorCache()

epoch = 10

cache.set(t,values)
for _ in range(epoch):
    start = time.time()
    ret_values = cache.get(t[:200])
    if ret_values is None:
        print("no hit")
        print(type(values))
        cache.set(t, values)
    else:
        print("hit")
        print(ret_values)
        print(ret_values["distance"].shape)

    print("elapsed: ", time.time()-start)