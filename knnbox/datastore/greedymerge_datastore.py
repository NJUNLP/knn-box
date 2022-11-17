import os
import json
import numpy as np
import time
import random
import tqdm

from knnbox.common_utils import Memmap, read_config, write_config 
from knnbox.datastore.utils import build_faiss_index, load_faiss_index 
from knnbox.datastore import Datastore


class GreedyMergeDatastore(Datastore):
    r"""
    implement greedy merge datastore
    """
    def prune(self, merge_neighbors = 2, batch_size = 4096, verbose = True):
        r"""
        prune the datastore using greedy merge strategy.
        """

        print("[Start Prune The Datastore]")
        start = time.time()
        # collect keys' neighbors first
        neighbors = self._collect_neighbors(merge_neighbors, batch_size, verbose)
        weights = np.memmap(os.path.join(self.path, "weights.npy"), dtype=int, mode="w+", shape=(self["vals"].size,))

        # init
        weights[:] = 1

        random_order = list(range(self["vals"].size))
        random.shuffle(random_order)

        for i, id_ in tqdm.tqdm(enumerate(random_order)):
            # if already removed
            if weights[id_] <= 0:
                continue

            for k, v in enumerate(neighbors[id_]):
                if id_ != v and weights[v] == 1 and self["vals"].data[v] == self["vals"].data[id_]:
                    # transfer the neighbors' weight to current entry
                    weights[v] = 0 
                    weights[id_] += 1
        
        
        # create new keys and values based on weights
        pruned_datastore_size = int((weights > 0).sum())
        if verbose:
            print(f"  > pruned datastore has {pruned_datastore_size} entries, \
                old datasotere has {self.values.capacity} entries,  \
                compress ratio: {float(pruned_datastore_size)/float(self.values.capacity)}")
        
        print("  > delete old datastore and construct pruned datastore...")
        self["new_keys"] = Memmap(os.path.join(self.path, "new_keys,npy"), mode="w+")
        self["new_vals"] = Memmap(os.path.join(self.path, "new_vals.npy"), mode="w+")

        # use memmap here
        del weights
        weights = Memmap(os.path.join(self.path, "weights.npy"), mode="r", dtype=int, shape= (self["vals"].size,))
        
        cnt = 0
        for i, wgh in enumerate(weights.data):
            if v > 0:
                self["new_keys"].data[cnt] = self["keys"].data[i]
                self["new_vals"].data[cnt] = self["vals"].data[i]
                weights.data[cnt] = v
                cnt += 1
        
        # delete old keys and values. and rename new_keys to keys, new_values to values.
        self["keys"] = self["new_keys"]
        self["vals"]= self["new_vals"]
        self["weights"] = weights
        del self["new_keys"]
        del self["new_vals"]

        os.remove(os.path.join(self.path, "keys.npy"))
        os.remove(os.path.join(self.path, "vals.npy"))
        os.remove(os.path.join(self.path, "neighbors_"+str(merge_neighbors)+".npy"))
        os.rename(os.path.join(self.path, "new_keys.npy"), os.path.join(self.path, "keys.npy"))
        os.rename(os.path.join(self.path, "new_vals.npy"), os.path.join(self.path, "vals.npy"))
        self.keys.filename =  os.path.join(self.path, "keys.npy")
        self.values.filename = os.path.join(self.path, "vals.npy")
        print("prune the datastore took {} s".format(time.time()-start))
        print("[Finished Pruning Datastore ^_^]")



    def _collect_neighbors(self, merge_neighbors = 2, batch_size = 4096, verbose=True):
        r"""
        collect the neighbors of original datastore's entry
        
        Args:
            merge_neighbors: merge how many neighbors
        """ 
        if not hasattr(self, "faiss_index") or self.faiss_index is None:
            self.load_faiss_index()
             
        # drop redundant space first
        self["keys"].drop_redundant()
        self["vals"].drop_redundant()

        neighbors = np.memmap(os.path.join(self.path, f"neighbors_{merge_neighbors}.npy"), dtype=np.int32, mode="w+", shape=(
            self["vals"].size, merge_neighbors+1)) # here we plus 1, because the nearest one always be the query entry self.
        
        if verbose:
            print("  > start collecting neighbors...")
            start_time = time.time()
        
        batches = []
        cnt = 0
        offset = 0
        for i in tqdm.tqdm(range(0, self["vals"].capacity)):
            if i % 100000 == 0:
                print(f"  > collecting {i}th entries")

            batches.append(self["keys"].data[i])
            cnt += 1
        
            if cnt % batch_size == 0 or i == self["vals"].size - 1:
                dists, knns = self.faiss_index.search(np.array(batches).astype(np.float32), merge_neighbors+1) # plus 1 as well
                neighbors[offset:offset+knns.shape[0]] = knns
                cnt = 0
                batches = []
                offset += knns.shape[0]
            
        if verbose:
            print(f"  > collect neighbors took {time.time()- start_time} seconds.")

        return neighbors

            
    