import json
import os
import faiss
import numpy as np
import time
import torch

global_datastores = {}

def get_registered_datastore(name):
    if name in global_datastores:
        return global_datastores[name]
    else:
        return None

def registe_datastore(name, datastore):
    global_datastores[name] = datastore


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
        json.dump(config, f)



def build_faiss_index(keys,
                shape,
                output_filename,
                train_index_count = 1000000,
                n_centroids = 4096,
                code_size = 64,
                n_probe = 32,
                num_keys_to_add_at_a_time = 500000,
                seed = 1,
                verbose=False
                ):
    r""" 
    this function is mostly inspired by adaptive-knn-mt code 
    """
    
    res = faiss.StandardGpuResources()
    capacity, dimension = shape

    if not os.path.exists(output_filename+".trained"):
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFPQ(quantizer, dimension, n_centroids, code_size, 8)
        index.nprobe = n_probe

        if verbose:
            print("Start put index to GPU")
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index, co)
        if verbose:
            print("Training index")
        np.random.seed(seed)
        random_sample = np.random.choice(
            np.arange(capacity), size = [min(train_index_count, capacity)],
            replace=False
        )
        start = time.time()
        if verbose:
            print(random_sample[:10])
            print(keys[random_sample][:10])
        # faiss dosent handle train keys in fp16, so convert to fp32 first
        gpu_index.train(keys[random_sample].astype(np.float32))
        if verbose:
            print("Training took {} s".format(time.time() - start))
            print("Writing index after training")
        faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), output_filename+".trained")
        if verbose:
            print("writing index took {} s".format(time.time() -start))
    
    if verbose:
        print("Adding Keys")
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
                print("Added %d tokens so far" % start)
            faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), output_filename)
    
    if verbose:
        start = time.time()
        faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), output_filename)
        if verbose:
            print("Adding total %d keys " % end)
            print("Adding took {} s".format(time.time() - start_time))


def load_faiss_index(path, shape, n_probe,
            move_to_gpu=False, verbose=False):
    r"""
    load the faiss index"""
    if verbose:
        start_time = time.time()
    
    index = faiss.read_index(path, faiss.IO_FLAG_ONDISK_SAME_DIR)
    if move_to_gpu:
        if verbose:
            print("move faiss index to gpu")
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        index = faiss.index_cpu_to_gpu(res, 0, index, co)
    if verbose:
        print("reading index took {} s".format(time.time()-start_time))
        print("the datastore shape is ", shape)
    index.nprobe = n_probe

    return index



def retrieve_k_nearest(query, faiss_index, k):
    r"""
    use faiss to retrieve k nearest item
    """
    query_shape = list(query.size())

    # TODO: 下面本来是view的，但是变成了reshape
    distances, indices = faiss_index.search(
                        query.detach().cpu().reshape(-1,query_shape[-1]).numpy(),
                        k
                        )
    
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
    tokens = tokens.unsqueeze(-1)
    return tokens, mask


def keys_mask_select(keys, mask):
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


