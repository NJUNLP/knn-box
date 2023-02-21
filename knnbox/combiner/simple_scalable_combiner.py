import torch
import torch.nn.functional as F

from knnbox.combiner.utils import calculate_knn_prob, calculate_combined_prob

class SimpleScalableCombiner:
    r"""
    A Combiner used by simple and scalable knn-mt
    """

    def __init__(self, temperature, probability_dim):
        self.lambda_ = None
        self.temperature = temperature
        self.probability_dim = probability_dim

    def get_knn_prob(self, vals, distances, temperature=None, device="cuda:0", **kwargs):
        r"""
        calculate knn prob for vanilla knn-mt
        parameter temperature will suppress self.parameter
        """
        temperature = temperature if temperature is not None else self.temperature  

        scaled_dists = - distances / temperature
        # use the paper formulation to get a dynamic lambda
        min_distance, _ = distances.min(dim=-1, keepdim=True) 
        self.lambda_ = torch.nn.functional.relu(1 - min_distance / self.temperature)
    
        knn_weights = torch.softmax(scaled_dists, dim=-1)
        B, S, K = vals.size()

        # construct prob
        knn_probs = torch.zeros(B, S, self.probability_dim, device=device)
        knn_probs.scatter_add_(dim=-1, index=vals, src=knn_weights)

        return knn_probs

    
    def get_combined_prob(self, knn_prob, neural_model_logit, lambda_ = None, log_probs = False):
        r""" 
        strategy of combine probability of vanilla knn-mt
        If parameter `lambda_` is given, it will suppress the self.lambda_ 
        """
        lambda_ = lambda_ if lambda_ is not None else self.lambda_
        return calculate_combined_prob(knn_prob, neural_model_logit, lambda_, log_probs)
        