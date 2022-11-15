from ..utils import retrieve_k_nearest
import torch

class Retriever:
    def __init__(self, datastore, k):
        self.datastore = datastore
        self.k = k
        self.results = None


    def retrieve(self, query, return_list = ["vals", "distances"], k = None ):
        r""" retrieve the datastore and return results 
        
        if parameter k is provided, it will suppress self.k
        """

        k = self.k if k is None else k
        # load the faiss index if haven't loaded
        if not hasattr(self.datastore, "faiss_index") or \
            self.datastore.faiss_index is None or "keys" not in self.datastore.faiss_index:
            self.datastore.load_faiss_index("keys", move_to_gpu=True)
        
        results = retrieve_k_nearest(query, self.datastore.faiss_index["keys"], self.k if k is None else k)
        distances = results["distances"]
        indices = results["indices"].cpu().numpy()
        
        ret = {}
        if "keys" in return_list:
            retrieved_keys = self.datastore["keys"].data[indices]
            ret["keys"] = torch.tensor(retrieved_keys, device=query.device, dtype=query.dtype)
        if "query" in return_list:
            ret["query"] = query

        retrieved_values = self.datastore["vals"].data[indices]
        if "vals" in return_list:
            ret["vals"] = torch.tensor(retrieved_values, device=query.device)
        if "indices" in return_list:
            ret["indices"] = results["indices"]
        if "distances" in return_list:
            ret["distances"] = distances
        if "k" in return_list:
            ret["k"] = k

        # if we have save the related sentence idx and token idx, return it
        if "sentence_ids" in return_list:
            assert "sentence_ids" in self.datastore.datas, "You must load the sentence_ids of datastore first."
            retrieved_sentence_ids = self.datastore["sentence_ids"].data[indices]
            ret["sentence_ids"] = torch.tensor(retrieved_sentence_ids, device=query.device)
        
        if "token_positions" in return_list:
            assert "token_positions" in self.datastore.datas, "You must lod the token_positions of datastore first"
            retrieved_token_positions = self.datastore["token_positions"].data[indices]
            ret["token_positions"] = torch.tensor(retrieved_token_positions, device=query.device)
        
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
        