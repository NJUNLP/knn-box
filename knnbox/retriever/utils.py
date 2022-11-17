r""" some utils function used for retrieve """
import torch

def retrieve_k_nearest(query, faiss_index, k):
    r"""
    use faiss to retrieve k nearest item
    """
    query_shape = list(query.size())

    # TODO: i dont know why can't use view but must use reshape here 
    distances, indices = faiss_index.search(
                        query.detach().cpu().float().reshape(-1,query_shape[-1]).numpy(), k)
    
    distances = torch.tensor(distances, device=query.device).view(*query_shape[:-1], k)
    indices = torch.tensor(indices,device=query.device).view(*query_shape[:-1], k)

    return {"distances": distances, "indices": indices}
