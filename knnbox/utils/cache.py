r""" implement a simple cache for tensor """
import torch
import numpy

class TensorCache():
    def __init__(self):
        self.cache = {}
    
    def get(self, keys):
        # x is a tensor, [a,....,z]
        original_shape = keys.shape
        keys = keys.view(-1, keys.size(-1))
        datas = []
        for idx, key in enumerate(keys):
            key_hash = self._hash(key)
            if key_hash not in self.cache:
                return None
            found = False
            for value in self.cache[key_hash]:
                if value["key"].equal(key):
                    datas.append(value)
                    found = True
                    break
            if found is False:
                return None

        ret = {}
        value_names = datas[0].keys()
        for name in value_names:
            buf = []
            for value in datas:
                buf.append(value[name])
            ret[name] = torch.stack(buf).squeeze(-1).view(*original_shape[:-1],-1) 
        return ret


    def set(self, keys, values):
        # x is a tensor, [a,...,z]
        keys = keys.view(-1, keys.size(-1))

        data_unit = {}
        for name, content in values.items():
            data_unit[name] = torch.tensor(content).view(-1, content.size(-1))

        # add to  
        for idx, key in enumerate(keys):
            key_hash = self._hash(key)
            unit = {"key": key}
            for name, data in data_unit.items():
                unit[name] = data[idx]
            if key_hash not in self.cache:
                self.cache[key_hash] = [unit]
            else:
                self.cache[key_hash].append(unit)

    def _hash(self, x):
        # x must be 1-d
        select_idx = [0,3,5,7,11,13,20]
        values = []
        for idx in select_idx:
            values.append(str(numpy.round(x[idx].data, 5)))
        
        hash_str = "".join(values)

        return hash_str
