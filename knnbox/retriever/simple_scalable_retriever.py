import torch
import numpy as np

import elasticsearch
import elasticsearch.helpers as helpers

from knnbox.common_utils import global_vars

class SimpleScalableRetriever:
    def __init__(self, k):
        self.k = k
        self.results = None

    
    def retrieve(self, query, encoder_out_hash, return_list = ["vals", "distances"], k = None ):
        r""" retrieve from dynamic datastore
        Args:
            query: [batch, 1, dim]
            encoder_out_hash: [batch]
        """
        k = self.k if k is None else k
        ret = {}
        batch_size, seq_len, dim  = query.shape # seq_len equals 1
        if "keys" in return_list:
            ret["keys"] = torch.empty(batch_size, seq_len, k, dim, dtype=query.dtype, device=query.device)
        if "vals" in return_list:
            ret["vals"] = torch.empty(batch_size, seq_len, k, dtype=torch.int64, device=query.device)
        if "distances" in return_list:
            ret["distances"] = torch.empty(batch_size, seq_len, k, dtype=torch.float, device=query.device)

        for sentence_idx, sentence in enumerate(query):
            for pos, q in enumerate(sentence):
                try:
                    dynamic_datastore = global_vars()["encoderout_to_kv"][float(encoder_out_hash[sentence_idx])]
                except:
                    # It is almost impossible to go to this branch,
                    # The main concern here is that using float as a key may lead to map misses
                    print("---- WARNING: Map miss ----")
                    nearest_dis = 1e5
                    nearest_key = ""
                    for h in global_vars()["encoderout_to_kv"].keys():
                        if abs(float(encoder_out_hash[sentence_idx]) - h) < nearest_dis:
                            nearest_dis = abs(float(encoder_out_hash[sentence_idx]) - h) 
                            nearest_key = h
                    dynamic_datastore = global_vars()["encoderout_to_kv"][nearest_key]
                
                if dynamic_datastore is not None:
                    # because dynamic datastore is small, we dont use faiss
                    distances = torch.cdist(q.unsqueeze(0), dynamic_datastore["keys"])
                    distances = distances.squeeze(0)
                    sorted_distances, indices = torch.sort(distances)

                    if indices.shape[0] < k:
                        # If the search result is less than k items (but not empty), 
                        # we can add some items with a very large distance
                        print("---- WARNING: Here a sentence retrieve less than ", k, " result ----")
                        sorted_distances = torch.cat(
                            (sorted_distances, 100000*torch.ones(k - sorted_distances.shape[0], dtype=torch.float, device=query.device)))
                        indices = torch.cat(
                            (indices, torch.zeros(k - indices.shape[0], dtype=torch.int64, device=query.device)))
                        
                    # only retain k items
                    if "keys" in return_list:
                        ret["keys"][sentence_idx][pos] = dynamic_datastore["keys"][indices[:k]]
                    if "vals" in return_list:
                        ret["vals"][sentence_idx][pos] = dynamic_datastore["vals"][indices[:k]]
                    if "distances" in return_list:
                        ret["distances"][sentence_idx][pos] = torch.square(sorted_distances[:k])
                else:
                    # if elasticsearch return empty result,
                    # we return dummy `keys` and `vals` with very big `distance`
                    if "keys" in return_list:
                        ret["keys"][sentence_idx][pos] = torch.zeros(k, dim, dtype=query.dtype, device=query.device)
                    if "vals" in return_list:
                        ret["vals"][sentence_idx][pos] = torch.zeros(k, dtype=torch.int64, device=query.device)
                    if "distances" in return_list:    
                        ret["distances"][sentence_idx][pos] = 100000 * torch.ones(k, dtype=torch.float, device=query.device)

        
        self.results = ret
        
        return ret



        
       