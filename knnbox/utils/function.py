import json
import os
import faiss
import numpy as np
import time
import torch

_global_vars = {}

def global_vars():
    return _global_vars

global_datastores = {}


def read_config(path):
    r"""
    read the config file under the `path` folder

    Args:
        path:
            folder where the config file is stored
    
    Returns:
        dict
    """
    config_file = os.path.join(path, "config.json")
    with open(config_file, encoding="utf-8", mode="r") as f:
        return json.load(f)


def write_config(path, config):
    r"""
    write the config file to the `path` folder

    Args:
        path:
            folder where the config file is stored
    
    Returns:
        dict
    """
    with open(os.path.join(path, "config.json"), encoding="utf-8", mode="w") as f:
        json.dump(config, f, indent = 6)


def build_faiss_index(
                keys,
                shape,
                output_filename,
                train_index_count = 1000000,
                n_centroids = 4096,
                code_size = 64,
                n_probe = 32,
                num_keys_to_add_at_a_time = 500000,
                seed = 1,
                use_pca = False,
                pca_dim = 256, # if use_pca==True, reduce to pca_dim before faiss retrieve
                verbose=False
                ):
    r""" 
    this function is mostly inspired by adaptive-knn-mt code 
    """
    print("[Start Building Faiss Index]")
    res = faiss.StandardGpuResources()
    capacity, dimension = shape

    progress_idx = 1
    total_progress = 4 if use_pca is False else 5

    if not os.path.exists(output_filename+".trained"):
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFPQ(quantizer, dimension, n_centroids, code_size, 8)
        index.nprobe = n_probe

        if verbose:
            print("  > [{}/{}] start put index to GPU...".format(progress_idx, total_progress))
        start = time.time()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index, co)
        if verbose:
            print("  > [{}/{}] put index to GPU took {} s". \
                format(progress_idx, total_progress, time.time()-start))
            progress_idx += 1

        # if use PCA, wrap the index with pre-PCA operation
        if use_pca == True:
            print("  > [{}/{}] do pca operation".format(progress_idx, total_progress))
            start = time.time()
            pca_matrix = faiss.PCAMatrix(dimension, pca_dim, 0, True)
            index = faiss.IndexPreTransform(pca_matrix, index)
            if verbose:
                print("  > [{}/{}] pca operation took {} s".\
                    format(progress_idx, total_progress, time.time()-start))
                progress_idx += 1

        if verbose:
            print("  > [{}/{}] training index (about 4 minutes)...".format(progress_idx, total_progress))
        start = time.time()
        np.random.seed(seed)
        random_sample = np.random.choice(
            np.arange(capacity), size = [min(train_index_count, capacity)],
            replace=False
        )
        # if verbose:
        #     print(random_sample[:10])
        #     print(keys[random_sample][:10])
        # faiss dosent handle train keys in fp16, so convert to fp32 first
        gpu_index.train(keys[random_sample].astype(np.float32))
        if verbose:
            print("  > [{}/{}] training took {} s".format(progress_idx, total_progress, time.time() - start))
            progress_idx += 1
            print("  > [{}/{}] writing index after training...".format(progress_idx, total_progress))
        start = time.time()
        faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), output_filename+".trained")
        if verbose:
            print("  > [{}/{}] writing index took {} s".format(progress_idx, total_progress, time.time() -start))
            progress_idx += 1
    
    if verbose:
        print("  > [{}/{}] adding keys...".format(progress_idx, total_progress))
    # read the trained model
    index = faiss.read_index(output_filename + ".trained")
    co = faiss.GpuClonerOptions()
    co.useFloat16 = True
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index, co)
    start = 0
    start_time = time.time()

    while start < capacity:
        end = min(capacity, start+num_keys_to_add_at_a_time)
        to_add = keys[start:end].copy()
        gpu_index.add_with_ids(to_add.astype(np.float32), np.arange(start,end))
        start += num_keys_to_add_at_a_time

        if (start % 1000000) == 0:
            if verbose:
                print("  > [{}/{}] added {} tokens so far, total {}.".format(
                            progress_idx, total_progress,min(start, capacity), capacity))
            faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), output_filename)
    
    if verbose:
        start = time.time()
        faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), output_filename)
        if verbose:
            print("  > [{}/{}] adding total {} keys ".format(progress_idx, total_progress, end))
            print("  > [{}/{}] adding took {} s".format(progress_idx, total_progress, time.time() - start_time))

    # remove the temporary trained index
    if os.path.exists(output_filename+".trained"):
        os.remove(output_filename+".trained")
    print("[Finish Building Faiss Index  Successfully ^_^]")

def load_faiss_index(path, shape, n_probe,
            move_to_gpu=False, verbose=False):
    r"""
    load the faiss index"""
    print("[Start Loading Faiss Index]")
    if verbose:
        start_time = time.time()
    
    # check if the faiss index has been built
    if not os.path.exists(path):
        print("!!Error: faiss index hasn't beed built, Pleast built it first and then load it")
        import sys
        sys.exit(1)
 
    index = faiss.read_index(path, faiss.IO_FLAG_ONDISK_SAME_DIR)
    if move_to_gpu:
        if verbose:
            print("  > move faiss index to gpu")
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        index = faiss.index_cpu_to_gpu(res, 0, index, co)
    if verbose:
        print("  > reading index took {} s".format(time.time()-start_time))
        print("  > the datastore shape is ", shape)
    index.nprobe = n_probe
    print("[Finish Loading Faiss Index For Successfully ^_^]")
    return index



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


def filter_pad_tokens(tokens, pad_idx=1):
    r"""
    given a int tensor, 
    return all no pad element and the mask,
    1 represent no-pad, 0 represent pad
    """
    mask = tokens.ne(pad_idx)
    tokens = tokens.masked_select(mask)
    return tokens, mask



def select_keys_with_pad_mask(keys, mask):
    r"""
    use the mask to chose keys 

    Args:
        keys: (batch_sz, seq, dim)
        mask: (batch_sz, seq)
    
    Return: (*, dim)
    """
    mask_shape = mask.size()
    mask = mask.unsqueeze(-1).repeat(*([1]*len(mask_shape)+[keys.size(-1)]))
    return keys.masked_select(mask).view(-1, keys.size(-1))



def disable_model_grad(model):
    r""" disable whole model's gradient """
    for name, param in model.named_parameters():
        param.requires_grad = False


def enable_module_grad(model, module_name):
    r""" enable a module's gridient caclulation by module name"""
    for name, param in model.named_parameters():
        if module_name in name:
            param.requires_grad = True

def label_smoothed_nll_loss(lprobs, target, epsilon=2e-3, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, 
