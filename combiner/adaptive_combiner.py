import torch
from torch import nn
import torch.nn.functional as F
import math
import os


class AdaptiveCombiner(nn.Module):
    def __init__(self, probability_dim, model = None, max_k = 32):
        super().__init__()
        if model is None:
            # use meta K network
            self.model = MetaKNetwork(max_k=max_k, temperature_trainable=False, temperature=10)
        else:
            self.model = model
        
        # make it gpu
        self.model.cuda()
        self.max_k = max_k
        self.temperature = 10 # TODO: 这一块应该整成参数?
        self.probability_dim = probability_dim
        self.lambda_ = None
        self.knn_prob = None


    def get_knn_prob(self, distances, values, device, **kwargs):

        # TODO: next line should not exist
        values = values.squeeze(-1)

        net_outputs = self.model(distances=distances, values=values)
        k_prob = net_outputs

        # # TODO: 切片改成更通用的形式，因为不一定有三个维度
        lambda_ = 1.0 - k_prob[:, :, 0:1] 
        k_soft_prob = k_prob[:,:,1:]
        # TODO: 计算knn prob，实现该函数
        knn_prob = self._caculate_select_knn_prob(values, distances, self.temperature, k_soft_prob, device)
        self.lambda_ = lambda_
        self.knn_prob = knn_prob

        return knn_prob



    def get_combined_prob(self, knn_prob, neural_model_logit, log_probs=False):
        # the policy to combine knn_prob and neural_model_prob
        neural_model_prob = F.softmax(neural_model_logit, dim=-1)
        combined_probs = knn_prob * self.lambda_ + neural_model_prob * (1 - self.lambda_)
        if log_probs:
            combined_probs =  torch.log(combined_probs)
        return combined_probs
    


    def _caculate_select_knn_prob(self, values, distances, temperature, knn_select_prob, device):
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
        scaled_dists = -distances / temperature
        knn_weight = torch.softmax(scaled_dists, dim=-1)  # [B, S, R_K, K]
        weight_sum_knn_weight = torch.matmul(knn_select_prob.unsqueeze(-2), knn_weight).squeeze(-2).unsqueeze(-1)  # [B, S, K, 1]
        knn_tgt_prob = torch.zeros(B, S, K, self.probability_dim).to(device)  # [B, S, K, Vocab Size]
        values = values.unsqueeze_(-1)  # [B, S, K, 1]

        knn_tgt_prob.scatter_(src=weight_sum_knn_weight.float(), index=values, dim=-1)
        prob = knn_tgt_prob.sum(dim=-2)  # [Batch Size, seq len, vocab size]

        return prob


    @staticmethod
    def load(path):
        r"""
        load an AdaptiveCombiner from disk
        """
        
        with open(os.path.join(path, "model.pt"), "rb") as f:
            state_dict = torch.load(f)
        model = MetaKNetwork()
        model.load_state_dict(state_dict)
        return Adaptivecombiner(model)  
    
    
    def dump(self, path):
        r"""
        save the AdaptiveCombiner to disk
        """
        # TODO:  config写入
        # create folder if not exist
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))



class MetaKNetwork(nn.Module):
    r""" meta k network in knn-mt """
    
    def __init__(self, 
                max_k = 32,
                k_trainable = True,
                lambda_trainable=True,
                temperature_trainable=True,
                label_count_as_feature=True,
                k_lambda_net_hid_size=32,
                k_lambda_net_drop_rate=0.0,
                relative_label_count=False,
                device="cuda:0",
                **kwargs,
                ):

        super().__init__()
        self.max_k = max_k
        self.k_trainable = k_trainable
        self.lambda_trainable = lambda_trainable
        self.temperature_trainable = temperature_trainable
        self.label_count_as_feature = label_count_as_feature
        self.k_lambda_net_hid_size = k_lambda_net_hid_size
        self.k_lambda_net_drop_rate = k_lambda_net_drop_rate
        self.lambda_ = kwargs.get("lambda_", None)
        self.temperature = kwargs.get("temperature", None)
        self.k = kwargs.get("k", None)
        self.relative_label_count = relative_label_count
        self.mask_for_label_count = None


        
        if self.k_trainable and self.lambda_trainable:
            self.retrieve_result_to_k_and_lambda = nn.Sequential(
                nn.Linear(max_k if not label_count_as_feature else max_k*2, k_lambda_net_hid_size),
                nn.Tanh(),
                nn.Dropout(p=self.k_lambda_net_drop_rate),
                nn.Linear(k_lambda_net_hid_size, 2+int(math.log(max_k, 2))), #[0, 1, 2, 4, 8 ..., max_k]
                nn.Softmax(dim=-1) 
            )

            # param init
            nn.init.xavier_normal_(self.retrieve_result_to_k_and_lambda[0].weight[:, : self.max_k], gain=0.01)
            if self.label_count_as_feature:
                nn.init.xavier_normal_(self.retrieve_result_to_k_and_lambda[0].weight[:,self.max_k:], gain=0.01)
        


    
    def forward(self, distances, values):
        if self.label_count_as_feature:
            label_counts = self._get_label_count_segment(values, relative=self.relative_label_count)
            network_inputs = torch.cat((distances.detach(), label_counts.detach().float()), dim=-1)
        else:
            network_inputs = distances.detach()

        if self.temperature_trainable:
            knn_temperature = None
        else:
            knn_temperature = self.temperature
        
        net_outputs = self.retrieve_result_to_k_and_lambda(network_inputs)
        
        # 该网络返回的是probs即可
        return net_outputs


    def _get_label_count_segment(self, values, relative=False):
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
            self.mask_for_label_count = mask_for_label_count.to(values.device)

        ## TODO: 感觉下面的特征不太对劲
        B, S, K = values.size()
        expand_values = values.unsqueeze(-2).expand(B,S,K,K)
        expand_values = expand_values.masked_fill(self.mask_for_label_count, value=-1)
        

        labels_sorted, _ = expand_values.sort(dim=-1) # [B, S, K, K]
        labels_sorted[:, :, :, 1:] *= ((labels_sorted[:, :, :, 1:] - labels_sorted[:, :, : , :-1]) != 0).long()
        retrieve_label_counts = labels_sorted.ne(0).sum(-1)
        # TODO: 下面这句,因为前面的都算了-1，但-1不应计算，所以去掉1
        retrieve_label_counts[:, :, :-1] -= 1

        if relative:
            relative_label_counts[:, :, 1:] = relative_label_counts[:, :, 1:] - relative_label_counts[:, :, :-1]
        
        return retrieve_label_counts





     


        