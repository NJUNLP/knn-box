from knnbox.combiner import Combiner
from knnbox.combiner.utils import calculate_knn_prob_with_merge_weight

class MergeWeightCombiner(Combiner):
    r""" 
    used by greedy merge knn-mt [when enable_cache=False, use_merge_weight=True]
    """
    def get_knn_prob(self,
                    vals,
                    distances,
                    merge_weights,
                    device="cuda:0",
                    **kwargs):
        return calculate_knn_prob_with_merge_weight(
            vals, distances, merge_weights, self.probability_dim,
            self.temperature, device, **kwargs )

    