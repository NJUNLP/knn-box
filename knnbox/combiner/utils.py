r""" some utils function used for combiner """

import torch
import torch.nn.functional as F

def calculate_knn_prob(vals, distances, probability_dim, temperature, device, **kwargs):
    r"""
    How vanilla knn-mt calculates knn probs using retrieved vals and distances.
    """
    scaled_dists = - distances / temperature
    knn_weights = torch.softmax(scaled_dists, dim=-1).unsqueeze(-1)
    
    probabilities_shape = list(vals.size()) + [probability_dim]

    # construct prob
    knn_probs = torch.zeros(*probabilities_shape, device=device)
    knn_probs.scatter_(dim=-1, index=vals.unsqueeze(-1), src=knn_weights)
    
    # sum same token's prob
    knn_probs = knn_probs.sum(dim=-2)

    return knn_probs


def calculate_combined_prob(knn_prob, neural_model_logit, lambda_, log_probs):
    r""" 
    How vanilla knn-mt calculate the combining probability.
    """
    neural_model_prob = F.softmax(neural_model_logit, dim=-1)
    combined_probs = knn_prob * lambda_ + neural_model_prob * (1 - lambda_)

    # some extra infomation
    extra = {}
    extra["neural_probs"] = neural_model_prob
    extra["unlog_combined_probs"] = combined_probs

    if log_probs:
        combined_probs =  torch.log(combined_probs)
    return combined_probs, extra