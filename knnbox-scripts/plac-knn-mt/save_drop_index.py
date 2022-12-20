from knnbox.datastore.datastore import Datastore
import argparse
import torch
from tqdm import tqdm
from knnbox.retriever import Retriever

if __name__ == "__main__":
    parser = argparse.ArgumentParser("save_drop_index")
    parser.add_argument("--plac-datastore-path", type=str, help="datastore path")
    parser.add_argument("--plac-k", type=int, metavar="N", default=4,
                        help="The hyper-parameter k_p of PLAC")
    parser.add_argument("--plac-bsz", type=int, metavar="N", default=4096,
                        help="The batch size of PLAC KM retrieval")
    
    args = parser.parse_args()    
    

    # load plac datastore (a vanilla datastore with mt_preds)
    datastore   = Datastore.load(args.plac_datastore_path, 
                load_list=["keys", "vals", "mt_preds"])
    # PLAC query excludes 1-NN from k_p, because 1-NN is query itself
    retriever   = Retriever(datastore=datastore, k=args.plac_k + 1)
    
    print(f"[ loaded mt_pred and vals ]", flush=True)
    
    # calc mt_known (mt_pred == target)
    datastore["mt_known"].add((datastore["mt_preds"].data == datastore["vals"].data).astype(int))
    
    print(f"[ mt_known calculated ]", flush=True)
    
    # query params
    query_size          = datastore["keys"].size
    query_batch_size    = args.plac_bsz
    
    
    for i in tqdm(range(0, query_size, query_batch_size)):
        start_idx   = i
        end_idx     = min(query_size, i + query_batch_size)

        query       = torch.from_numpy(datastore["keys"].data[start_idx:end_idx, :]).cuda()
        
        mt_known    = retriever.retrieve(query=query, return_list=["mt_known"])["mt_known"]
        known_count = torch.sum(mt_known, dim=-1)
        
        # 1-NN should be known, and the other k_p-NNs as well
        drop_index  = torch.nonzero(known_count == args.plac_k + 1).squeeze()
        datastore[f"drop_index_k{args.plac_k}"].add(drop_index + start_idx)
    
    drop_count = datastore[f"drop_index_k{args.plac_k}"].size
    
    print(f"[ {drop_count}/{query_size} ({drop_count/query_size:.4f}) of datastore can be pruned \^o^/ ]", flush=True)
    datastore.dump()
