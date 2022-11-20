r""" 
A Combiner used with CacheRetriever.
Firstly used by greedy-merge knn-mt
"""
import torch
import torch.nn.functional as F

from knnbox.combiner.utils import (
    calculate_knn_prob,
    calculate_combined_prob,
    calculate_knn_prob_with_merge_weight
)

class CacheCombiner:
    r"""
    Combiner use with CacheRetriever.
    """

    def __init__(self, lambda_, temperature, probability_dim):
        self.lambda_ = lambda_
        self.temperature = temperature
        self.probability_dim = probability_dim

    def get_knn_prob(
                    self,
                    cache,
                    query,
                    vals,
                    distances, 
                    query_idx_which_use_cache,
                    query_idx_which_use_datastore,
                    cached_probs,
                    original_query_shape,
                    merge_weights=None, # support for greedy_merge
                    device="cuda:0",
                    **kwargs,
                    ):
        r""" get knn probs.
        for those query which use cache, directly use cached probabilty.
        for those query which use datastore, calculate the probabilty with vals and distances.
        """

        # check
        assert query.size(0) == vals.size(0), "Error"
        assert distances.size(0) == vals.size(0), "Error"

        if merge_weights is not None:
            # support for greedy merge knn-mt
            datastore_retrieved_probs = calculate_knn_prob_with_merge_weight(
                vals, distances, merge_weights, self.probability_dim, self.temperature, device)
        else: 
            datastore_retrieved_probs = calculate_knn_prob(vals, 
                        distances, self.probability_dim, self.temperature, device)
        
        probabilities_shape = list(original_query_shape[:-1]) + [self.probability_dim]
        knn_probs = torch.zeros(*probabilities_shape, device=device)
        knn_probs = knn_probs.view(-1, self.probability_dim)


        if query_idx_which_use_cache.numel() > 0:
            knn_probs[query_idx_which_use_cache] = cached_probs
        if query_idx_which_use_datastore.numel() > 0:
            knn_probs[query_idx_which_use_datastore] = datastore_retrieved_probs
        knn_probs = knn_probs.view(*probabilities_shape)


        # update retriever cache
        if query_idx_which_use_datastore.numel() > 0:
            if cache["queries"] is None:
                cache["queries"] = query
                cache["probs"] = datastore_retrieved_probs
            else:
                cache["queries"] = torch.cat((cache["queries"], query), dim=0)
                cache["probs"] = torch.cat((cache["probs"], datastore_retrieved_probs), dim=0)

        return knn_probs


    def get_combined_prob(self, knn_prob, neural_model_logit, log_probs=False):
        return calculate_combined_prob(knn_prob, neural_model_logit, self.lambda_, log_probs)

