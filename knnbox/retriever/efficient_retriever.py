# Retriever for efficient knn-mt
# Efficient has a cache. when EfficientRetriever retrieve, It preferentially queries from the cache. 
# If there are no entries close enough in the cache, it queries from the datastore

import torch
from knnbox.retriever.utils import retrieve_k_nearest

class EfficientRetriever:
    def __init__(self, datastore, k, enable_cache=False, knn_cache_threshold=6):
        self.datastore = datastore
        self.k = k
        self.results = None
        self.enable_cache = enable_cache
        if enable_cache:
            self.knn_cache = None
            self.knn_cache_probs = None
            self.knn_cache_threshold = knn_cache_threshold


    def cache_knn_probs(self, knn_probs):
        if self.knn_cache_probs is None:
            self.knn_cache_probs = probs
        else:
            self.knn_cache_probs = torch.cat([self.knn_cache_probs, knn_probs], dim = 0)

    def retrieve(self, query, start_of_sentence=False, return_keys=False, return_query=False):
        r""" retrieve the datastore and return results """

        if not hasattr(self.datastore, "faiss_index") or self.datastore.faiss_index is None:
            self.datastore.load_faiss_index(move_to_gpu=True)

        if self.enable_cache:
            mask = torch.ones(query.size(0), dtype=torch.bool) # if mask == True, use datastore to retreve. otherwise, use cache.
            cache_probs = torch.empty()
            # when a new sentence comes, clear the cache
            if start_of_sentence:
                self.knn_cache = None
                self.knn_cache_probs = None
            
            # min distance of every query wtih knn_cache
            if self.knn_cache is not None:
                dists = torch.cdist(query.squeeze(1), self.knn_cache.squeeze(1), p=2).min(-1) 
                self.knn_cache = torch.cat([self.knn_cache, x], dim=0)
                use_cache_indices = (dists.values <= self.knn_cache_threshold).nonzero()[:,0]
                mask[use_cache_indices] = False
                query = query[mask]

                if indices.size(0) > 0:
                    use_cache_probs[indices] = self.knn_cache_probs[dists.indices[use_cache_indices]] 
            else:
                self.knn_cache = query            

        results = retrieve_k_nearest(query, self.datastore.faiss_index, self.k)
        distances = results["distances"]
        indices = results["indices"].cpu().numpy()
        
        ret = {}
        if return_keys:
            retrieved_keys = self.datastore.keys.data[indices]
            ret["keys"] = torch.tensor(retrieved_keys, device=query.device, dtype=query.dtype)
        if return_query:
            ret["query"] = query

        retrieved_values = self.datastore.values.data[indices]
        ret["values"] = torch.tensor(retrieved_values, device=query.device)
        ret["indices"] = torch.tensor(indices, device=query.device)
        ret["distances"] = distances.to(query.device)
        # the weight of pruned datastore entries
        ret["weights"] = torch.tensor(self.datastore.weights.data[indices], device=query.device)
        ret["k"] = self.k
        if self.enable_cache:
            ret["use_cache_indices"] = use_cache_indices
            ret["use_cache_probs"] =  use_cachce_probs
        
        self.results = ret
        return ret
   
    
    @staticmethod
    def load(path):
        """
        load a retriever from disk
        """
        pass

    def save(path):
        r"""
        save a retriever to disk
        """
        pass
        
