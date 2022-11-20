r""" 
A retriever with cache, firstly used by greedy merge knn-mt.
The retriever clear it's cache when a new batch comes """

import torch
from knnbox.retriever.utils import retrieve_k_nearest

class CacheRetriever:

    def __init__(self, datastore, k):
        self.datastore = datastore
        self.k = k
        self.results = None
        # The cache
        self.cache = {"queries": None, "probs": None}


    def retrieve(self, query, return_list = ["keys", "vals", "distances"], cache_threshold=6.0):
        r"""
        retrieve the datastore and results with a cache. 
        note: for those queries which use cache, only return it's cached probs
        """
        # load the faiss index if haven't loaded
        if not hasattr(self.datastore, "faiss_index") or \
               self.datastore.faiss_index is None or "keys" not in self.datastore.faiss_index:
            self.datastore.load_faiss_index("keys")
        
        ret = {}
        query = query.detach()
        ret["original_query_shape"] = query.size()
        ret["cache"] = self.cache

        # flatten the query
        query = query.view(-1, query.size(-1)) # [XX, dim]

        # if cache is not empty, check which query can use cache to retrieve
        if self.cache["queries"] is not None:
            # calculate the Euclidean distance between current queries and the currents on cache 
            distance_matrix = torch.cdist(query, self.cache["queries"], p=2) # [XX, cache_size]
            min_distance, min_indices = distance_matrix.min(dim=-1)
            # if distances smaller than threshold, directly return probs
            mask = min_distance <= cache_threshold
            query_idx_which_use_cache = mask.nonzero(as_tuple=True)[0]
            cached_probs = self.cache["probs"][min_indices[query_idx_which_use_cache]]
            ret["query_idx_which_use_cache"] = query_idx_which_use_cache
            ret["cached_probs"] = cached_probs
            # split queries which need to use datastore.
            ret["query_idx_which_use_datastore"] = (~mask).nonzero(as_tuple=True)[0]
            query_using_datastore = query[ret["query_idx_which_use_datastore"]]
        
        # if cache is empty, all query use datastore to retrieve
        else:
            ret["query_idx_which_use_cache"] = torch.empty(0)
            ret["cached_probs"] = torch.empty(0)
            ret["query_idx_which_use_datastore"] = torch.arange(start=0, end=query.size(0), device=query.device)
            query_using_datastore = query

        # use datastore to handle quires which cant use cache
        query = query_using_datastore
        faiss_results = retrieve_k_nearest(query, self.datastore.faiss_index["keys"], self.k)
        if "distances" in return_list:
            ret["distances"] = faiss_results["distances"]
        if "indices" in return_list:
            ret["indices"] = faiss_results["indices"]
        if "k" in return_list:
            ret["k"] = k
        if "query" in return_list:
            ret["query"] = query

        # other information get from self.datastores.datas using indices, for example `keys` and `vals`
        indices = faiss_results["indices"].cpu().numpy()
        for data_name in return_list:
            if data_name not in ["distances", "indices", "k", "query"]:
                assert data_name in self.datastore.datas, \
                                    "You must load the `{}` of datastore first".format(data_name)
                ret[data_name] = torch.tensor(self.datastore[data_name].data[indices], device=query.device)
        
        self.results = ret # save the retrieved results
        return ret


    def clear_cache(self):
        r""" clear the cache """
        self.cache["queries"] = None
        self.cache["vals"] = None
        

