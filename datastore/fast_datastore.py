""" fast datastore to implement Fast Knn-MT """
import os

class FastDatastore:
    r"""
    FastDatastore is to implement Fast knn-mt.
    """
    def __init__(
        self,
        path,
        key_dim = 768,
        value_dim = 1,
        key_dtype = "memmap_float32",
        value_dtype = "memmap_int",
        shards = None
    ):
        if shards is None:
            self.shards = {}
        else:
            self.shards = shards

        self.mask = None
            
    
    def add_key(key, shard_name):
        # if the shard datastore does not exist, create one
        if shard_name not in self.shards:
            self.shards[shard_name] = Datastore(
                os.path.join(path, shard_name),
                key_dim = self.key_dim,
                value_dim = self.value_dim,
                key_dtype = self.key_dtype,
                value_dtype = self.value_dtype
            )
        # add keys to shard datastore
        self.shards[shard_name].add_key(key)

    
    def add_value(value, shard_name):
        # if not exist, create one
        if shard_name not in self.shards:
            self.shards[shard_name] = Datastore(
                os.path.join(path, shard_name),
                key_dim = self.key_dim,
                value_dim = self.value_dim,
                key_dtype = self.key_dtype,
                value_dtype = self.value_dtype
            )
        # add values to shard datstore
        self.shards[shard_name].add_value(value)

    
    @staticmethod
    def load(path):
        config = read_config(path)
        shards = {}
        for shard_name in config["shard_names"]:
            shard_full_path = os.path.join(path, shard_name)
            shards[shard_name] = Datastore.load(shard_full_path)
        
        return FastDatastore(**config, shards=shards)


    def dump(self):
        # dump every datastore
        for shard in self.shards.values():
            shard.dump()
    

    def build_faiss_index():
        # build faiss index for every shard datastore
        for datastore in self.shards.values():
            datastore.build_faiss_index()
    