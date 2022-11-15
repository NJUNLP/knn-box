import torch
import torch.nn.functional as F


class Combiner:
    r"""
    A simple Combiner"""

    def __init__(self, lambda_, temperature, probability_dim):
        self.lambda_ = lambda_
        self.temperature = temperature
        self.probability_dim = probability_dim

    def get_knn_prob(self, distances, vals, device="cpu", temperature=None, **kwargs):
        r"""caculate the prob """

        vals_shape = list(vals.size())
        scaled_dists = -distances / (self.temperature if temperature is None else temperature)
        knn_weights = torch.softmax(scaled_dists, dim=-1).unsqueeze(-1)
        vals_shape.append(self.probability_dim)
        probabilities_shape = vals_shape

        # construct prob
        knn_probs = torch.zeros(*probabilities_shape, device=device)
        knn_probs.scatter_(dim=-1, index=vals.unsqueeze(-1), src=knn_weights)

        # sum same token's prob
        knn_probs = knn_probs.sum(dim=-2)
        return knn_probs

    
    def get_combined_prob(self, knn_prob, neural_model_logit, log_probs=False, lambda_=None):
        r""" 
        strategy of combine probability
        
        If parameter `lambda_` is given, it will suppress the self.lambda_ 
        """

        lambda_ = self.lambda_ if lambda_ is None else lambda_
        neural_model_prob = F.softmax(neural_model_logit, dim=-1)
        combined_probs = knn_prob * lambda_ + neural_model_prob * (1 - lambda_)

        # some extra infomation
        extra = {}
        extra["neural_probs"] = neural_model_prob
        extra["unlog_combined_probs"] = combined_probs

        if log_probs:
            combined_probs =  torch.log(combined_probs)
        return combined_probs, extra