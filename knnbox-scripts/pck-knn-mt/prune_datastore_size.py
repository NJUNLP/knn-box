import argparse
from shutil import copyfile
from knnbox.datastore.pck_datastore import PckDatastore


if __name__ == "__main__":
    parser = argparse.ArgumentParser("prune datastore size")
    parser.add_argument("--prune-style", choices=["random", 
                "prune_high_ppl","prune_low_ppl", "prune_half_low_half_high_ppl", "prune_similar_ppl",
                "prune_high_entropy", "prune_low_entropy", "prune_half_low_half_high_entropy","prune_similar_entropy"
                ], help="chose the prune policy")
    parser.add_argument("--prune-ratio", type=float, 
                help="retain how much entries.")
    parser.add_argument("--n-of-4-gram", type=int,
                help="you can use n-gram  less than 4")
    parser.add_argument("--min-sample-threshold", type=int,
                help="the minimum threshold to prune")
    parser.add_argument("--thread-num", type=int,
                help="use how many thread to prune the size")
    parser.add_argument("--input-datastore-path", type=str,
                help="the path of the datastore to be pruned")
    parser.add_argument("--output-datastore-path", type=str,
                help="the path to save the pruned datastore")
    args = parser.parse_args()


    datastore = PckDatastore.load(args.input_datastore_path,
                load_list=["keys", "vals", "ids_4_gram", "probs_4_gram", "entropy"], load_network=False)

    datastore.prune_size(args.output_datastore_path, n_of_4_gram=args.n_of_4_gram, prune_style=args.prune_style,
                minimum_sample=args.min_sample_threshold, sample_rate=args.prune_ratio,
                thread_num=args.thread_num)
