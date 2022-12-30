import torch
from torch import nn
import torch.nn.functional as F
import math
import os
from knnbox.common_utils import read_config, write_config
from knnbox.combiner.utils import calculate_combined_prob, calculate_knn_prob


class RobustCombiner(nn.Module):
    r""" Robust knn-mt Combiner """
    def __init__(self, 
                max_k,
                probability_dim,
                k_trainable = True,
                lambda_trainable = True,
                temperature_trainable = True,
                **kwargs
                ):
        super().__init__()
        self.meta_k_network = MetaKNetwork(max_k, 
                    k_trainable, lambda_trainable, temperature_trainable, **kwargs)
        
        self.max_k = max_k
        self.probability_dim = probability_dim
        self.k_trainable = k_trainable
        self.lambda_trainable = lambda_trainable
        self.temperature_trainable = temperature_trainable
        self.kwargs = kwargs 
        self.mask_for_distance = None

        # check 
        assert self.lambda_trainable or "lambda_" in kwargs, \
            "if lambda is not trainable, you should provide a fixed lambda_ value"
        assert self.temperature_trainable or "temperature" in kwargs, \
            "if temperature is not trainable, you should provide a fixed temperature"
        
        self.k = None if self.k_trainable else kwargs["k"]
        self.lambda_ = None if self.lambda_trainable else kwargs["lambda_"]
        self.temperature = None if self.temperature_trainable else kwargs["temperature"]

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        
        def _apply(m):
            if hasattr(m, "set_num_updates") and m != self:
                m.set_num_updates(num_updates)

        self.apply(_apply)

    def get_knn_prob(self, vals, keys, distances, net_output, target, last_hidden, device="cuda:0"):
        metak_outputs = self.meta_k_network(vals, distances, keys, net_output, target, last_hidden)
        # TODO 
        if self.lambda_trainable:
            self.lambda_ = metak_outputs["lambda_net_output"]
        
        if self.temperature_trainable:
            self.temperature = metak_outputs["temperature_net_output"]
        
        if self.k_trainable:
            # generate mask_for_distance just for once
            if not hasattr(self, "mask_for_distance") or self.mask_for_distance is None:
                self.mask_for_distance = self._generate_mask_for_distance(self.max_k, device)
            
            k_probs = metak_outputs["k_net_output"]
            B, S, K = vals.size()
            R_K = k_probs.size(-1)

            distances = distances.unsqueeze(-2).expand(B, S, R_K, K)
            distances = distances * self.mask_for_distance  # [B, S, R_K, K]
            if self.temperature_trainable:
                temperature = self.temperature.unsqueeze(-1).expand(B, S, R_K, K)
            else:
                temperature = self.temperature
            distances = - distances / temperature

            knn_weight = torch.softmax(distances, dim=-1)  # [B, S, R_K, K]
            weight_sum_knn_weight = torch.matmul(k_probs.unsqueeze(-2), knn_weight).squeeze(-2).unsqueeze(-1)  # [B, S, K, 1]
            knn_prob = torch.zeros(B, S, K, self.probability_dim, device=device)  # [B, S, K, Vocab Size]
            # construct the knn 
            knn_prob.scatter_(src=weight_sum_knn_weight.float(), index=vals.unsqueeze(-1), dim=-1)
            knn_prob = knn_prob.sum(dim=-2)  # [Batch Size, seq len, vocab size]

        else:
            # if k is not trainable, the process of calculate knn probs is same as vanilla knn-mt
            knn_prob = calculate_knn_prob(vals, distances, self.probability_dim,
                        self.temperature, device=device)

        return knn_prob
        
    
    def get_combined_prob(self, knn_prob, neural_model_logit, log_probs=False):
        r""" get combined probs of knn_prob and neural_model_prob """
        return calculate_combined_prob(knn_prob, neural_model_logit, self.lambda_, log_probs)


    def dump(self, path):
        r""" dump the robust knn-mt to disk """
        # step 1. write config
        config = {}
        config["max_k"] = self.max_k
        config["probability_dim"] = self.probability_dim
        config["k_trainable"] = self.k_trainable
        config["lambda_trainable"] = self.lambda_trainable
        config["temperature_trainable"] = self.temperature_trainable
        for k, v in self.kwargs.items():
            config[k] = v
        write_config(path, config)
        # step 2. save model
        torch.save(self.state_dict(), os.path.join(path, "robust_combiner.pt"))


    @classmethod
    def load(cls, path):
        r""" load the robust knn-mt from disk """
        config = read_config(path)
        robust_combiner = cls(**config)

        robust_combiner.load_state_dict(torch.load(os.path.join(path, "robust_combiner.pt")))
        return robust_combiner
    

    @staticmethod
    def _generate_mask_for_distance(max_k, device):
        k_mask = torch.empty((max_k, max_k)).fill_(999.)
        k_mask = torch.triu(k_mask, diagonal=1) + 1
        power_index = torch.tensor([pow(2, i) - 1 for i in range(0, int(math.log(max_k, 2)) + 1)])
        k_mask = k_mask[power_index]
        k_mask.requires_grad = False
        k_mask = k_mask.to(device)
        return k_mask



class MetaKNetwork(nn.Module):
    r""" meta k network of robust knn-mt """
    def __init__(
        self,
        max_k = 32,
        k_trainable = True,
        lambda_trainable = True,
        temperature_trainable = True,
        k_net_hid_size = 32,
        lambda_net_hid_size = 32,
        temperature_net_hid_size = 32,
        k_net_dropout_rate = 0.0,
        lambda_net_dropout_rate = 0.0,
        temperature_net_dropout_rate = 0.0,
        label_count_as_feature = True,
        relative_label_count = False,
        device = "cuda:0",
        **kwargs,
    ):
        super().__init__()
        self.max_k = max_k    
        self.k_trainable = k_trainable
        self.lambda_trainable = lambda_trainable
        self.temperature_trainable = temperature_trainable
        self.label_count_as_feature = label_count_as_feature
        self.relative_label_count = relative_label_count
        self.device = device
        self.mask_for_label_count = None
        self.mid_size = 32
        self.num_updates = 0

        # Robust kNN-MT always uses the same configuration
        self.distance_func = nn.Sequential(
            nn.Linear(self.max_k * 2 + 8, 1),
        )
        self.distance_fc1 = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )
        self.distance_fc2 = nn.Sequential(
            nn.Linear(self.max_k * 2, self.mid_size),
            nn.Tanh(),
            nn.Linear(self.mid_size, 2),
        )

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self.num_updates = num_updates

    def forward(self, vals, distances):
        # TODO
        if self.label_count_as_feature:
            label_counts = self._get_label_count_segment(vals, relative=self.relative_label_count)
            network_inputs = torch.cat((distances.detach(), label_counts.detach().float()), dim=-1)
        else:
            network_inputs = distances.detach()
        
        results = {}
        
        results["k_net_output"] = self.distance_to_k(network_inputs) if self.k_trainable else None
        results["lambda_net_output"] = self.distance_to_lambda(network_inputs) if self.lambda_trainable else None
        results["temperature_net_output"] = self.distance_to_temperature(network_inputs) \
                    if self.temperature_trainable else None
        
        return results
    

    def _get_label_count_segment(self, vals, relative=False):
        r""" this function return the label counts for different range of k nearest neighbor 
            [[0:0], [0:1], [0:2], ..., ]
        """
        
        # caculate `label_count_mask` only once
        if self.mask_for_label_count is None:
            mask_for_label_count = torch.empty((self.max_k, self.max_k)).fill_(1)
            mask_for_label_count = torch.triu(mask_for_label_count, diagonal=1).bool()
            mask_for_label_count.requires_grad = False
            # [0,1,1]
            # [0,0,1]
            # [0,0,0]
            self.mask_for_label_count = mask_for_label_count.to(vals.device)

        ## TODO: The feature below may be unreasonable
        B, S, K = vals.size()
        expand_vals = vals.unsqueeze(-2).expand(B,S,K,K)
        expand_vals = expand_vals.masked_fill(self.mask_for_label_count, value=-1)
        

        labels_sorted, _ = expand_vals.sort(dim=-1) # [B, S, K, K]
        labels_sorted[:, :, :, 1:] *= ((labels_sorted[:, :, :, 1:] - labels_sorted[:, :, : , :-1]) != 0).long()
        retrieve_label_counts = labels_sorted.ne(0).sum(-1)
        retrieve_label_counts[:, :, :-1] -= 1

        if relative:
            relative_label_counts[:, :, 1:] = relative_label_counts[:, :, 1:] - relative_label_counts[:, :, :-1]
        
        return retrieve_label_counts


