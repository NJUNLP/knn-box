import os
import json
from ..utils import Memmap, read_config, build_faiss_index, load_faiss_index, write_config 


class Datastore:
    r"""
    implement vanilla datastore for neural network
    """

    def __init__(
        self,
        path,
        key_dim = 768,
        value_dim = 1,
        key_dtype = "memmap_float16",
        value_dtype = "memmap_int",
        keys = None,
        values = None,
        has_keys = True,
        **kwargs,
    ):
        r"""
        Args:
            path(`str`):
                the directory to save datastore files
        
        """
        self.path = path
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.key_dtype = key_dtype
        self.value_dtype = value_dtype

        # create folder if not exist
        if not os.path.exists(path):
            os.mkdir(path)
        
        if keys is not None:
            self.keys = keys
        elif has_keys is True:
            if key_dtype.startswith("memmap"):
                print(key_dtype)
                key_dtype = key_dtype.split("_")[1]
                self.keys = Memmap(
                    os.path.join(path, "keys"),
                    dtype = key_dtype,
                    dim = key_dim,
                    mode = "w+",
                )
        else:
            self.keys = None
                
        if values is not None:
            self.values = values
        else:
            if value_dtype.startswith("memmap"):
                value_dtype = value_dtype.split("_")[1]
                self.values = Memmap(
                    os.path.join(path, "values"),
                    dtype = value_dtype,
                    dim = value_dim,
                    mode = "w+",
                )

        self.faiss_index = None
        self.mask = None

        

    def add_key(self, key):
        self.keys.add(key)

    
    def add_value(self, value):
        self.values.add(value)
    

    def set_mask(self, mask):
        self.mask = mask

    
    def get_mask(self):
        return self.mask
    

    @staticmethod
    def load(path, load_key=True):
        r"""
        load the datastore under the `path` folder

        Args:
            path(`str`):
                folder where the datastore files is stored
        Return:
            Datastore object(`Datastore`)
        """

        config = read_config(path)
        # load the memmap from disk
        keys_filename = os.path.join(path, "keys")
        has_keys = True
        if load_key is True and config["key_dtype"].startswith("memmap"):
            config["key_dtype"] = config["key_dtype"].split("_")[1]
            keys = Memmap(keys_filename, dim=config["key_dim"],
                    dtype=config["key_dtype"], mode="r",
                    capacity=config["capcity"])
        else:
            keys = None
            has_keys = False

        values_filename = os.path.join(path, "values")
        if config["value_dtype"].startswith("memmap"):
            config["value_dtype"] = config["value_dtype"].split("_")[1]
            values = Memmap(values_filename, dim=config["value_dim"],
                    dtype=config["value_dtype"], mode="r",
                    capacity=config["capcity"])
        
        # 创建datastore
        ds = Datastore(**config, keys=keys, values=values, has_keys=has_keys)
        return ds



    def load_faiss_index(self, move_to_gpu=True):
        r"""
        load faiss index, so you can use self.faiss_index
        """
        path = os.path.join(self.path, "keys.faiss")
        capcity = self.keys.capacity if self.keys is not None else self.values.capacity
        self.faiss_index = load_faiss_index(path,
        shape = (capcity, self.key_dim),
        n_probe = 32,
        move_to_gpu=move_to_gpu)
        


    
    def dump(self, verbose=True):
        r"""
        store the datastore files and config file to disk.

        
        Args:
            path(`str`):
                folder where you want to save the datastore files
        """
        self.keys.dump()
        self.values.dump()
        config = {}
        config["path"] = self.path
        config["key_dim"] = self.key_dim
        config["value_dim"] = self.value_dim
        config["key_dtype"] = self.key_dtype
        config["value_dtype"] = self.value_dtype
        config["capcity"] = self.values.capacity
        write_config(self.path, config)
        if verbose:
            print("keys: %d " % self.keys.capacity)
            print("values: %d " % self.values.capacity)
            assert self.keys.capacity == self.values.capacity, "keys entries not equal to values"
            print("added %d entries" % config["capcity"])



    def build_faiss_index(self, verbose=False):
        r"""
        build faiss index for keys.
        the output file named `keys.faiss`
        """

        if not isinstance(self.keys, Memmap):
            print("[ERROR]: can only build faiss for Memmap / np.NDArray")
            os.exit(1)
        
        # build faiss
        build_faiss_index(self.keys.data, 
                    self.keys.data.shape,
                    os.path.join(self.path, "keys.faiss"),
                    verbose=verbose)

 
