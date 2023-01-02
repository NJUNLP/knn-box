import argparse
import torch
import os
from shutil import copyfile
from knnbox.datastore import PckDatastore


if __name__ == "__main__":
    parser = argparse.ArgumentParser("reduct datastore dim")
    parser.add_argument("--input-datastore-path", type=str,
                        help="the datastore we want to reduct dimension")
    parser.add_argument("--output-datastore-path", type=str,
                        help="the output path to save the reducted datastore")
    args = parser.parse_args()
    datastore = PckDatastore.load(args.input_datastore_path, load_list=["keys", "vals"], load_network=True)
    
    # reduct keys
    new_keys = datastore.reconstruct_keys_with_reduction_network(output_dir=args.output_datastore_path)
    datastore.path = args.output_datastore_path
    datastore["keys"] = new_keys
    copyfile(os.path.join(args.input_datastore_path, "vals.npy"), 
        os.path.join(args.output_datastore_path, "vals.npy"))
    datastore.dump(dump_network=True)

    # build faiss
    torch.cuda.empty_cache() # release unused gpu
    datastore.build_faiss_index("keys") # build faiss index for reducted keys

                                    
