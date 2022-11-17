import torch
import torch.nn as nn

from knnbox.retriever.utils import retrieve_k_nearest

class SelectRetriever(nn.Module):
    def __init__(self, datastore, k, select_network, return_keys=False):
        self.datastore = datastore
        self.k = k
        self.return_keys = return_keys
        self.results = None
        # select network is used to select samples to retrieve
        self.select_network = select_network if select_network else SelectNetwork()

    def retrieve(self, query, return_keys=False):
        # load faiss_index if needed
        if self.datastore.faiss_index is None:
            self.datastore.load_faiss_index(move_to_gpu=True) 
        
        need_select = self.select_network(query)
        
        results = retrieve_k_nearest(query, self.datastore.faiss_index, self.k)
        distances = results["distances"]
        indices = results["indices"].cpu().numpy()
        
        ret = {}
        if return_keys:
            retrieved_keys = self.datastore.keys.data[indices]
            ret["keys"] = retrieved_keys

        retrieved_values = self.datastore.values.data[indices]
        ret["values"] = torch.tensor(retrieved_values, device=query.device)
        ret["indices"] = torch.tensor(indices, device=query.device)
        ret["distances"] = distances.to(query.device)
        ret["k"] = self.k
        self.results = ret
        return ret        
    
    @staticmethod
    def load(path):
        """
        load a select retriever from disk
        """
        ## TODO: load the 
        pass 

    def save(path):
        r"""
        save a retriever to disk
        """

# network for determin which sample need retrieve
class SelectNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.model(x)