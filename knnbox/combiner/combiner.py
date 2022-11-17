import torch
import torch.nn.functional as F

from knnbox.combiner.utils import calculate_knn_prob, calculate_combined_prob

class Combiner:
    r"""
    A simple Combiner used by vanilla knn-mt
    """

    def __init__(self, lambda_, temperature, probability_dim):
        self.lambda_ = lambda_
        self.temperature = temperature
        self.probability_dim = probability_dim

    def get_knn_prob(self, vals, distances, temperature=None, device="cuda:0", **kwargs):
        r"""
        calculate knn prob for vanilla knn-mt
        parameter temperature will suppress self.parameter
        """
        temperature = temperature if temperature is not None else self.temperature  
        return calculate_knn_prob(vals, distances, self.probability_dim,
                     temperature, device, **kwargs)

    
    def get_combined_prob(self, knn_prob, neural_model_logit, lambda_ = None, log_probs = False):
        r""" 
        strategy of combine probability of vanilla knn-mt
        If parameter `lambda_` is given, it will suppress the self.lambda_ 
        """
        lambda_ = lambda_ if lambda_ is not None else self.lambda_
        return calculate_combined_prob(knn_prob, neural_model_logit, lambda_, log_probs)
        

        