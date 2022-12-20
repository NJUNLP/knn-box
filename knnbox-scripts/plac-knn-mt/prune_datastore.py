from knnbox.datastore.datastore import Datastore
import argparse
import numpy as np
import torch
from tqdm import tqdm
from knnbox.retriever import Retriever

if __name__ == "__main__":
    parser = argparse.ArgumentParser("save_drop_index")
    parser.add_argument("--datastore-path", type=str, help="path to full datastore")
    parser.add_argument("--pruned-datastore-path", type=str, help="path to pruned datastore", default=None)
    parser.add_argument("--plac-datastore-path", type=str, help="path to plac datastore (with drop index)")
    parser.add_argument("--plac-k", type=int, metavar="N", default=4,
                        help="The hyper-parameter k_p of PLAC")
    parser.add_argument("--plac-ratio", type=float, metavar="R", default=1.0,
                        help="The hyper-parameter ratio of PLAC")
    parser.add_argument("--plac-bsz", type=int, metavar="N", default=4096,
                        help="The batch size of PLAC new datastore dump")
    parser.add_argument("--build-faiss-index-with-cpu", action="store_true", default=False,
                        help="use faiss-cpu instead of faiss-gpu (useful when gpu memory is small)") 
    
    args = parser.parse_args()    
    
    
    

    # load plac datastore (for drop index)
    plac_datastore      = Datastore.load(args.plac_datastore_path, 
                                         load_list=["vals", f"drop_index_k{args.plac_k}"])
    drop_index:np.memmap= plac_datastore[f"drop_index_k{args.plac_k}"].data
    print(f"[ loaded drop_index_k{args.plac_k} ]", flush=True)
    
    # load original datastore (only need keys and vals)
    datastore           = Datastore.load(args.datastore_path, 
                                         load_list=["keys", "vals"])
    datastore_size      = datastore["vals"].size
    print(f"[ loaded original datastore, size {datastore_size} ]", flush=True)

    # using size for a quick check
    # assert plac_datastore["vals"].size == datastore["vals"].size and np.equal(plac_datastore["vals"].data, datastore["vals"].data)
    assert plac_datastore["vals"].size == datastore["vals"].size, "The plac datastore must match the datastore to be pruned"
    
    # justify ratio
    prune_ratio         = round(args.plac_ratio, 2)
    max_ratio = round(drop_index.size / datastore["vals"].size, 2)
    if prune_ratio > max_ratio:
        print(f"The provided ratio [{prune_ratio:.2f}] is bigger than max ratio [{max_ratio:.2f}], using max ratio instead")
        prune_ratio         = max_ratio
    print(f"[ PLAC prune with k={args.plac_k} and ratio={prune_ratio:.2f} ]", flush=True)
    
    # declare pruned_datastore
    pruned_datastore    = Datastore(args.pruned_datastore_path or args.datastore_path + f'plac_k{args.plac_k}_ratio{prune_ratio:.2f}')
    
    # calculate keep index
    np.random.shuffle(drop_index)
    keep_index          = np.setdiff1d(np.arange(datastore_size), drop_index[:int(round(datastore_size * prune_ratio))], assume_unique=True)
    pruned_size         = keep_index.size
    print(f"[ pruned datastore will have size {pruned_size} ]", flush=True)
    
    # query params
    query_size          = keep_index.size
    query_batch_size    = args.plac_bsz
    
    # batch index select
    for i in tqdm(range(0, query_size, query_batch_size)):
        start_idx   = i
        end_idx     = min(query_size, i + query_batch_size)

        pruned_datastore["keys"].add(datastore["keys"].data[keep_index[start_idx:end_idx]])
        pruned_datastore["vals"].add(datastore["vals"].data[keep_index[start_idx:end_idx]])
    
    print(f"[ copied {pruned_size} keys and vals from original datastore, ready to build index ]", flush=True)
    
    # dump and build_index
    pruned_datastore.dump()
    pruned_datastore.build_faiss_index("keys", use_gpu=(not args.build_faiss_index_with_cpu))
    
    print(f"[ PLAC pruning successfully completed \^o^/ ]", flush=True)
