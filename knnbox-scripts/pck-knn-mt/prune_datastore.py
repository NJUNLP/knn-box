from knnbox.datastore.pck_datastore import PckDatastore, TripletDatastoreSamplingDataset
import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser("prune_datastore")
    parser.add_argument("--datastore-path", type=str, help="datastore path")
    parser.add_argument("--need-prune-size", action="store_true", default=False,
                        help="wheter to prune the size of datastore")
    parser.add_argument("--need-train-network", action="store_true", default=False,
                        help="whether to train the reduct network")
    parser.add_argument("--need-reduct-dimension", action="store_true", default=False,
                        help="whether to use the reduct network to reduct datastore keys dimension")
    parser.add_argument("--prune-style", choices=["random", 
                "prune_high_ppl","prune_low_ppl", "prune_half_low_half_high_ppl", "prune_similar_ppl",
                "prune_high_entropy", "prune_low_entropy", "prune_half_low_half_high_entropy","prune_similar_entropy"
                ], help="chose the prune policy")
    parser.add_argument("--thread-num", type=int, default=10, help="thread num for pruning")
    parser.add_argument("--n-of-4-gram", type=int, help="n of 4 gram")
    parser.add_argument("--prune-sample-rate", type=float, default=0.1,
            help="when prune, sample rate")
    parser.add_argument("--prune-minimum-sample", type=int, default=2,
            help="when prune, the minimum sample size to prune")
    parser.add_argument("--dataset-sample-rate", type=float, default=0.4,
            help="pck-knn-mt dataset sample rate when train reduct network")
    parser.add_argument("--train-batch-size", type=int, default=10,
            help="the batch size to train reduction network")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
            help="the learning rate when train reduct network")
    parser.add_argument("--epoch", type=int, default=10,
            help="train how much epoch")
    parser.add_argument("--dr-loss-ratio", type=float, default=0.3, help="loss ratio of dr")
    parser.add_argument("--nce-loss-ratio", type=float, default=0.3, help="loss ratio of nce")
    parser.add_argument("--wp-loss-ratio", type=float, default=0.4, help="loss ratio of wp")
    args = parser.parse_args()    

    datastore = PckDatastore.load(args.datastore_path, 
                load_list=["keys", "vals", "ids_4_gram", "probs_4_gram", "entropy"])
    # step 1. prune the size
    if args.need_prune_size:
        datastore.prune(n_of_4_gram=args.n_of_4_gram, prune_style=args.prune_style, 
            minimum_sample=args.prune_minimum_sample, sample_rate=args.prune_sample_rate,
            thread_num=args.thread_num)
        datastore.dump()

    # step 2. train reduction network
    if args.need_train_network:
        triplet_dataset = TripletDatastoreSamplingDataset(
            dictionary_len=datastore.dictionary_len,
            use_cluster=True,
            verbose=True,
            db_keys=datastore["keys"].data,
            db_vals=datastore["vals"].data,
            sample_rate=args.dataset_sample_rate,
        )
        datastore.train_reduction_network(
            triplet_dataset,
            args.train_batch_size,
            args.dr_loss_ratio,
            args.nce_loss_ratio,
            args.wp_loss_ratio,
            args.learning_rate,
            args.epoch,
        )
        datastore.dump()

    # step 3. reduct datastore keys with trained reduction network 
    if args.need_reduct_dimension:
        datastore.reconstruct_keys_with_reduction_network(batch_size=100)
        datastore.dump()

    # build faiss    
    if args.need_prune_size or args.need_reduct_dimension:
        torch.cuda.empty_cache()    # release unused GPU memory
        datastore.build_faiss_index("keys")
  