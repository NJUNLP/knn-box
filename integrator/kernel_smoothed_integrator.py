# learning kernel-smoothed machine translation with retrieved examples  


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from libds.utils.utils import read_config, write_config

class KernelSmoothedIntegrator(nn.Module):
    r"""
    kernel smoothed integrator"""

    def __init__(self, query_dim, probability_dim, bandwidth_estimator=None, weight_estimator=None, device='cuda:0', kernel_type='laplacian'):
        super().__init__()
        if bandwidth_estimator is None:
            self.bandwidth_estimator = BandwidthEstimator(query_dim=query_dim, device=device)
        else:
            self.bandwidth_estimator = bandwidth_estimator.to(device)
        
        if weight_estimator is None:
            self.weight_estimator = WeightEstimator(query_dim=query_dim,device=device)
        else:
            self.weight_estimator = weight_estimator.to(device)
        
        self.bandwidth_estimator.cuda()
        self.weight_estimator.cuda()

        self.query_dim = query_dim
        self.probability_dim = probability_dim
        self.kernel_type = kernel_type
        self.lambda_ = None



    def get_knn_prob(self, query, keys, distances, values, device="cpu", train_KSTER=False, **kwargs):
        r"""caculate the knn prob """
        # TODO: to fix
        values = values.squeeze(-1)

        # if we are training KSTER, drop the nearest key value pair
        if train_KSTER is True:
            keys = keys[...,1:,:]
            values = values[...,1:]
            distances = distances[...,1:]
    

        # caclulate bandwidth
        # keys shape: [..., k, probability_dim]
        values_shape = list(values.size())
        average_key = torch.mean(keys, dim=-2)
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
        
        values_shape.append(self.probability_dim)
        probabilities_shape = values_shape
        # construct prob
        knn_probs = torch.zeros(*probabilities_shape, device=device)
        knn_probs.scatter_(dim=-1, index=values.unsqueeze(-1), src=knn_weights)
        # sum up same tok's prob
        knn_probs = knn_probs.sum(dim=-2)

        # caculate the lambda
        self.lambda_ = self.weight_estimator(query, weighted_sum_key)
        return knn_probs

    @staticmethod
    def load(path):
        r"""
        load kernel smoothed integrator from disk"""
        config = read_config(path)
        print(config)
        query_dim = config["query_dim"]
        probability_dim = config["probability_dim"]
        kernel_type = config["kernel_type"]
        with open(os.path.join(path, "bandwidth_estimator.pt"), "rb") as f:
            bandwidth_estimator_state_dict = torch.load(f)
        with open(os.path.join(path, "weight_estimator.pt"), "rb") as f:
            weight_estimator_state_dict = torch.load(f)
        bandwidth_model = BandwidthEstimator(query_dim)
        weight_model = WeightEstimator(query_dim)
        bandwidth_model.load_state_dict(bandwidth_estimator_state_dict)
        weight_model.load_state_dict(weight_estimator_state_dict)
        return KernelSmoothedIntegrator(query_dim, probability_dim, 
            bandwidth_estimator=bandwidth_model, weight_estimator=weight_model,kernel_type=kernel_type)

    def dump(self, path):
        r"""
        dump a kernel smoothed integrator to disk"""
        config = {}
        config["query_dim"] = self.query_dim
        config["probability_dim"] = self.probability_dim
        config["kernel_type"] = self.kernel_type
        write_config(path, config)
        torch.save(self.bandwidth_estimator.state_dict(), os.path.join(path, "bandwidth_estimator.pt"))
        torch.save(self.weight_estimator.state_dict(), os.path.join(path, "weight_estimator.pt"))


    def get_integrated_prob(self, knn_prob, neural_model_logit, log_probs=False):
        r""" 
        strategy of combine probability """
        neural_model_prob = F.softmax(neural_model_logit, dim=-1)
        integrated_probs = knn_prob * self.lambda_ + neural_model_prob * (1 - self.lambda_)
        if log_probs:
            integrated_probs =  torch.log(integrated_probs)
        return integrated_probs



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
            nn.Linear(query_dim * 2, query_dim), # we set inner dimmension to 256
            nn.ReLU(),
            nn.Linear(query_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, query, weighted_sum_key):
        x = torch.cat((query, weighted_sum_key), dim=-1)
        return self.model(x)