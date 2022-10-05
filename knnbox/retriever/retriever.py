from ..utils import retrieve_k_nearest
import torch

class Retriever:
    def __init__(self, datastore, k):
        self.datastore = datastore
        self.k = k
        self.results = None


    def retrieve(self, query, return_keys=False, return_query=False):
        r""" retrieve the datastore and return results """

        if self.datastore.faiss_index is None:
            self.datastore.load_faiss_index(move_to_gpu=True)
         
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
        pass
        