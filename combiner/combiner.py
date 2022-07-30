import torch
import torch.nn.functional as F


class Combiner:
    r"""
    A simple Combiner"""

    def __init__(self, lambda_, temperature, probability_dim):
        self.lambda_ = lambda_
        self.temperature = temperature
        self.probability_dim = probability_dim

    def get_knn_prob(self, distances, values, device="cpu", **kwargs):
        r"""caculate the prob """

        # TODO: next line should not exist
        values = values.squeeze(-1)
        values_shape = list(values.size())

        scaled_dists = -distances / self.temperature
        knn_weights = torch.softmax(scaled_dists, dim=-1).unsqueeze(-1)

        values_shape.append(self.probability_dim)
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