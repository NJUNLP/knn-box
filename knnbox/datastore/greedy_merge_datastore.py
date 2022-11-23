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
        weights = np.memmap(os.path.join(self.path, "total_merge_weights.npy"), dtype=int, mode="w+", shape=(self["vals"].size,))

        # init
        weights[:] = 1

        random_order = list(range(self["vals"].size))
        random.shuffle(random_order)
        print("  > rebuild weights...")
        start = time.time()
        with tqdm.tqdm(total=len(random_order)) as pbar:
            for i, id_ in enumerate(random_order):
                pbar.update(1)
                # if already removed
                if weights[id_] <= 0:
                    continue
                for k, v in enumerate(neighbors[id_]):
                    if id_ != v and weights[v] == 1 and self["vals"].data[v] == self["vals"].data[id_]:
                        # transfer the neighbors' weight to current entry
                        weights[v] = 0 
                        weights[id_] += 1
        print("  > rebuild weights took {} seconds.".format(time.time()-start))
        
        # create new keys and values based on weights
        pruned_datastore_size = int((weights > 0).sum())
        if verbose:
            print(f"  > pruned datastore has {pruned_datastore_size} entries, "
                  f" old datasotere has {self['vals'].size} entries,  "
                  f"compress ratio: {float(pruned_datastore_size) / float(self['vals'].size)}")
        
        print("  > delete old datastore and construct pruned datastore...")
        start = time.time()
        self["new_keys"] = Memmap(os.path.join(self.path, "new_keys.npy"), mode="w+")
        self["new_vals"] = Memmap(os.path.join(self.path, "new_vals.npy"), mode="w+")
        self["merge_weights"] = Memmap(os.path.join(self.path, "merge_weights.npy"), mode="w+")
        
        with tqdm.tqdm(total=weights.shape[0]) as pbar:
            for i, wgh in enumerate(weights):
                pbar.update(1)
                if wgh > 0:
                    self["new_keys"].add(self["keys"].data[i].reshape(1,-1))
                    self["new_vals"].add(self["vals"].data[i].reshape(1))
                    self["merge_weights"].add(wgh)

        
        # delete old keys and values. and rename new_keys to keys, new_values to values.
        self["keys"] = self["new_keys"]
        self["vals"]= self["new_vals"]
        del self["new_keys"]
        del self["new_vals"]
        del weights

        os.remove(os.path.join(self.path, "keys.npy"))
        os.remove(os.path.join(self.path, "vals.npy"))
        os.remove(os.path.join(self.path, "total_merge_weights.npy"))
        os.remove(os.path.join(self.path, "neighbors_"+str(merge_neighbors)+".npy"))

        os.rename(os.path.join(self.path, "new_keys.npy"), os.path.join(self.path, "keys.npy"))
        os.rename(os.path.join(self.path, "new_vals.npy"), os.path.join(self.path, "vals.npy"))
        self["keys"].filename = os.path.join(self.path, "keys.npy")
        self["vals"].filename = os.path.join(self.path, "vals.npy")
        print("construct pruned datastore took {} s".format(time.time()-start))
        print("[Finished Pruning Datastore ^_^]")



    def _collect_neighbors(self, merge_neighbors = 2, batch_size = 4096, verbose=True):
        r"""
        collect the neighbors of original datastore's entry
        
        Args:
            merge_neighbors: merge how many neighbors
        """ 
        if not hasattr(self, "faiss_index") or self.faiss_index is None:
            self.load_faiss_index("keys", verbose=False)
             
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
        for i in tqdm.tqdm(range(0, self["vals"].size)):
            batches.append(self["keys"].data[i])
            cnt += 1
        
            if cnt % batch_size == 0 or i == self["vals"].size - 1:
                dists, knns = self.faiss_index["keys"].search(np.array(batches).astype(np.float32), merge_neighbors+1) # plus 1 as well
                neighbors[offset:offset+knns.shape[0]] = knns
                cnt = 0
                batches = []
                offset += knns.shape[0]

        # release memory
        del self.faiss_index["keys"] 
        if verbose:
            print(f"  > collect neighbors took {time.time()- start_time} seconds.")
        
        return neighbors

            
    