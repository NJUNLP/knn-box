import torch
import torch.nn.functional as F
from .adaptive_combiner import AdaptiveCombiner
from .kernel_smoothed_combiner import KernelSmoothedCombiner


class GreedyMergeCombiner:
    r"""
    Greedy Merge knn-mt Combiner"""

    def __init__(self, lambda_, temperature, probability_dim):
        self.lambda_ = lambda_
        self.temperature = temperature
        self.probability_dim = probability_dim

    def get_knn_prob(self, distances, values, use_weight=True, device="cpu", **kwargs):
        r"""caculate the prob """

        # if use cache
        if "use_cache_indices" in kwargs:
            use_cache_indices = kwargs["use_cache_indices"]
            use_cachce_probs = kwargs["use_cache_probs"]

        values = values.squeeze(-1)
        values_shape = list(values.size())
        values_shape.append(self.probability_dim)

        scaled_dists = -distances / self.temperature + (torch.log(kwargs["weights"].float()).squeeze(-1) if use_weight else 0.0)
        
        knn_weights = torch.softmax(scaled_dists, dim=-1).unsqueeze(-1)

        probabilities_shape = values_shape
        # construct prob
        knn_probs = torch.zeros(*probabilities_shape, device=device)
        knn_probs.scatter_(dim=-1, index=values.unsqueeze(-1), src=knn_weights)

        # sum same tok's prob
        knn_probs = knn_probs.sum(dim=-2)
        return knn_probs

    
    def get_combined_prob(self, knn_prob, neural_model_logit, log_probs=False):
        r""" 
        strategy of combine probability """
        neural_model_prob = F.softmax(neural_model_logit, dim=-1)
        combined_probs = knn_prob * self.lambda_ + neural_model_prob * (1 - self.lambda_)
        if log_probs:
            combined_probs =  torch.log(combined_probs)
        return combined_probs


class EfficientAdaptiveCombiner(AdaptiveCombiner):
     # combine efficient knn-mt with adaptive knn-mt
    def get_knn_prob(self, distances, values, device, **kwargs):
        
        # TODO: next line should not exist
        values = values.squeeze(-1)

        net_outputs = self.model(distances=distances, values=values)
        k_prob = net_outputs

        # # TODO: 切片改成更通用的形式，因为不一定有三个维度
        lambda_ = 1.0 - k_prob[:, :, 0:1] 
        k_soft_prob = k_prob[:,:,1:]
        # TODO: 计算knn prob，实现该函数
        knn_prob = self._caculate_select_knn_prob(values, distances, self.temperature, k_soft_prob, device, kwargs["weights"])
        self.lambda_ = lambda_
        self.knn_prob = knn_prob

        return knn_prob
    def _caculate_select_knn_prob(self, values, distances, temperature, knn_select_prob, device, weights):
        r""" using k select prob to caculate knn prob """
        B, S, K = distances.size()
        R_K = knn_select_prob.size(-1)

        # caculate mask for distance if not exist
        if hasattr(self, "mask_for_distance") is False:
            k_mask = torch.empty((self.max_k, self.max_k)).fill_(999.)
            k_mask = torch.triu(k_mask, diagonal=1) + 1

            power_index = torch.tensor([pow(2, i) - 1 for i in range(0, int(math.log(self.max_k, 2)) + 1)])
            k_mask = k_mask[power_index]

            k_mask.requires_grad = False
            k_mask = k_mask.to(device)
            self.mask_for_distance = k_mask
        
        distances = distances.unsqueeze(-2).expand(B, S, R_K, K)
        distances = distances * self.mask_for_distance
        scaled_dists = -distances / temperature + torch.log(weights.float()).squeeze(-1)
        knn_weight = torch.softmax(scaled_dists, dim=-1)  # [B, S, R_K, K]
        weight_sum_knn_weight = torch.matmul(knn_select_prob.unsqueeze(-2), knn_weight).squeeze(-2).unsqueeze(-1)  # [B, S, K, 1]
        knn_tgt_prob = torch.zeros(B, S, K, self.probability_dim).to(device)  # [B, S, K, Vocab Size]
        values = values.unsqueeze_(-1)  # [B, S, K, 1]

        knn_tgt_prob.scatter_(src=weight_sum_knn_weight.float(), index=values, dim=-1)
        prob = knn_tgt_prob.sum(dim=-2)  # [Batch Size, seq len, vocab size]

        return prob
    


class EfficientKernelSmoothedCombiner(KernelSmoothedCombiner):
    # combine efficient kernel
    # overwrite get_knn_probs to add weights when caclulate scaled_dists
    def get_knn_prob(self, query, keys, distances, values, device="cpu", train_KSTER=False, **kwargs):
        r"""caculate the knn prob """

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
            scaled_dists = - distances / bandwidth + torch.log(kwargs["weights"].float()).squeeze(-1)
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