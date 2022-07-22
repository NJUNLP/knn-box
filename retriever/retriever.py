from ..utils import retrieve_k_nearest
from ..utils.cache import TensorCache
import torch

class Retriever:
    def __init__(self, datastore, k, return_keys=False, enable_cache=False):
        self.datastore = datastore
        self.k = k
        self.return_keys = return_keys
        self.results = None
        self.enable_cache = enable_cache   
        self.cache = TensorCache() if enable_cache else None


    def retrieve(self, query, return_keys=False, return_query=False):
        if self.datastore.faiss_index is None:
            self.datastore.load_faiss_index(move_to_gpu=True)
        
        # if self.enable_cache:
        #     cache_results = self.cache.get(query)
        #     if cache_results is None:
        #         results = retrieve_k_nearest(query, self.datastore.faiss_index, self.k)
        #         distances = results["distances"]
        #         indices = results["indices"].cpu().numpy()
        #         content = {"distances": distances, "indices": indices}
        #         self.cache.set(query, content)
        #     else:
        #         distances = cache_results["distances"]
        #         indices = cache_results["indices"]
        # else:
        results = retrieve_k_nearest(query, self.datastore.faiss_index, self.k)
        distances = results["distances"]
        indices = results["indices"].cpu().numpy()
        
        ret = {}
        if return_keys:
            retrieved_keys = self.datastore.keys.data[indices]
            ret["keys"] = torch.tensor(retrieved_keys, device=query.device)
        if return_query:
            ret["query"] = query

        retrieved_values = self.datastore.values.data[indices]
        ret["values"] = torch.tensor(retrieved_values, device=query.device)
        ret["indices"] = torch.tensor(indices, device=query.device)
        ret["distances"] = distances.to(query.device)
        ret["k"] = self.k
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
        