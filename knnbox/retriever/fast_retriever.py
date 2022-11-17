import numpy as np
from knnbox.retriever import Retriever
from knnbox.datastore import Datastore
from knnbox.retriever.utils import retrieve_k_nearest

class FastRetriever:
    def __init__(self, k, shard_datastore):
        self.shard_datastore = shard_datastore
        self.k = k
        self.dynamic_datastore = None
        self.src_tokens = None
        self.src_token_vectors = None

    def set_tokens(tokens):
        self.src_tokens = tokens


    def set_token_vectors(token_vectors):
        self.src_token_vectors = token_vectors


    def retrieve(self, query):
        r"""
        retrieve dynamic datastore 
        """
        self.build_dynamic_datastore()
        ## TODO: implement matrix k nearest
        result = matrix_k_nearest(query, self.dynamic_datastore, k)
        distances = result["distances"]
        indices = result["indices"]

        ret = {}
        if return_keys:
            retrieved_keys = self.dynamic_datastore.keys.data[indices]
            ret["keys"] = retrieved_keys

        retrieved_values = self.dynamic_datastore.values.data[indices]
        ret["values"] = torch.tensor(retrieved_values, device=query.device)
        ret["indices"] = torch.tensor(indices, device=query.device)
        ret["distances"] = torch.tensor(distances, device=query.device)
        ret["k"] = self.k
        return ret


    def build_dynamic_datastore(self): 
        r"""
        build dynamic datastore for retrieve,
        """
        self.dynamic_datastore = Datastore(
            "",
            key_dim=shard_datastore.key_dim,
            value_dim=shard_datastore.value_dim,
            key_dtype="ndarray_float32",
            value_dtype="ndarray_int"
        )
        
        # use src_tokens and its vectors to chose entries
        for token, vector in zip(self.src_tokens, self.src_token_vectors):
            result = retrieve_k_nearest(vector, self.shard_datastore[token], self.k)
            indices = result["indices"]
            self.dynamic_datastore.add_key(self.shard_datastore.keys[indices])
            self.dynamic_datastore.add_value(self.shard_datastore.values[indices])
        

        
        

