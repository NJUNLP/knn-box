import torch
import numpy as np

import elasticsearch
import elasticsearch.helpers as helpers

from knnbox.common_utils import global_vars

class SimpleScalableRetriever:
    def __init__(self, elastic_address, elastic_index_name, k):
        self.k = k
        # connect to es
        self.es = elasticsearch.Elasticsearch([elastic_address], request_timeout=3600)
        self.es_index_name = elastic_index_name
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

        query = query.squeeze(1)
        for idx, q in enumerate(query):
            try:
                dynamic_datastore = global_vars()["encoderout_to_kv"][float(encoder_out_hash[idx])]
            except:
                nearest_dis = 1e5
                nearest_key = ""
                for h in global_vars()["encoderout_to_kv"].keys():
                    if abs(float(encoder_out_hash[idx]) - h) < nearest_dis:
                        nearest_dis = abs(float(encoder_out_hash[idx]) - h) 
                        nearest_key = h
                dynamic_datastore = global_vars()["encoderout_to_kv"][nearest_key]
                print("--------------------fuck----------------------")
            # because dynamic datastore is small, we dont use faiss
            distances = torch.cdist(q.unsqueeze(0), dynamic_datastore["keys"])
            distances = distances.squeeze(0)
            sorted_distances, indices = torch.sort(distances)
            # only retain k entry
            if "keys" in return_list:
                keys = dynamic_datastore["keys"][indices[:k]]
                ret["keys"][idx][0] = keys 
            if "vals" in return_list:
                vals = dynamic_datastore["vals"][indices[:k]]
                ret["vals"][idx][0] = vals
            if "distances" in return_list:
                ret["distances"][idx][0] = torch.square(sorted_distances[:k])
        
        self.results = ret
        
        return ret
            



        
       