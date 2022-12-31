import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from knnbox.common_utils import read_config, write_config
from knnbox.combiner.utils import calculate_combined_prob

class KernelSmoothedCombiner(nn.Module):
    r"""
    combiner for kernel smoothed knn-mt
    """
    def __init__(self, query_dim, probability_dim, device='cuda:0', kernel_type='laplacian'):
        super().__init__()

        self.bandwidth_estimator = BandwidthEstimator(query_dim=query_dim, device=device)
        self.weight_estimator = WeightEstimator(query_dim=query_dim, device=device)
        
        self.device = device 
        self.query_dim = query_dim
        self.probability_dim = probability_dim
        self.kernel_type = kernel_type
        self.lambda_ = None



    def get_knn_prob(self, query, keys, vals, distances, device="cuda:0", **kwargs):
        r"""caculate the knn prob """
        # if we are training KSTER, drop the nearest key value pair
        if self.training:
            keys = keys[...,1:,:]
            vals = vals[...,1:]
            distances = distances[...,1:]
    
        # caclulate bandwidth
        # keys shape: [..., k, probability_dim]
        average_key = torch.mean(keys, dim=-2)
        # query and average_key may be half precision, convert them to float32 first
        query = query.float()
        average_key = average_key.float()
        # bandwidth, i.e. temperature
        bandwidth = self.bandwidth_estimator(query, average_key) # [..., k]

        # caclulate knn_probs
        # distance [..., k], kernel_out [..., k]
        if self.kernel_type == 'gaussian':
            scaled_dists = - distances / bandwidth
        else:
            scaled_dists = - torch.sqrt(distances) / bandwidth
        
        knn_weights = torch.softmax(scaled_dists, dim=-1).unsqueeze(-1)
        
        weighted_sum_key = knn_weights.repeat(*([1]*(knn_weights.dim()-1)), keys.size(-1)) * keys
        weighted_sum_key = torch.sum(weighted_sum_key, dim=-2)
        
        B, S, K = vals.size()
        # construct prob
        knn_probs = torch.zeros(B, S, self.probability_dim, device=device)
        knn_probs.scatter_add_(dim=-1, index=vals, src=knn_weights.squeeze(-1))

        # caculate the lambda
        self.lambda_ = self.weight_estimator(query, weighted_sum_key)
        return knn_probs



    def get_combined_prob(self, knn_prob, neural_model_logit, log_probs=False):
        r""" 
        strategy of combine probability 
        """
        return calculate_combined_prob(knn_prob, neural_model_logit, self.lambda_, log_probs) 


    def dump(self, path):
        r"""
        dump a kernel smoothed combiner to disk"""
        # step 1. dump the config
        if not os.path.exists(path):
            os.makedirs(path)
        config = {}
        config["query_dim"] = self.query_dim
        config["probability_dim"] = self.probability_dim
        config["kernel_type"] = self.kernel_type
        write_config(path, config)
        # step 2. dump the model
        torch.save(self.state_dict(), os.path.join(path, "kernel_smoothed_combiner.pt"))


    @classmethod
    def load(cls, path):
        r"""
        load kernel smoothed combiner from disk"""
        # step 1. load the config
        config = read_config(path)
        kernel_smoothed_combiner = cls(**config)

        # step 2. load the model state dict
        kernel_smoothed_combiner.load_state_dict(
                torch.load(os.path.join(path, "kernel_smoothed_combiner.pt")))
         
        return kernel_smoothed_combiner



class BandwidthEstimator(nn.Module):
    def __init__(self, query_dim, device='cuda:0'):
        super().__init__()
        self.fc = nn.Linear(query_dim * 2, 1)
    
    def forward(self, query, average_key):
        # concatenate the query and average_k
        x = torch.cat((query, average_key), dim=-1)
        x = self.fc(x)
        x = torch.exp(x)
        return x


class WeightEstimator(nn.Module):
    r""" model to get the lamba weight"""
    def __init__(self, query_dim, device='cuda:0'):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(query_dim * 2, query_dim),
            nn.ReLU(),
            nn.Linear(query_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, query, weighted_sum_key):
        x = torch.cat((query, weighted_sum_key), dim=-1)
        return self.model(x)