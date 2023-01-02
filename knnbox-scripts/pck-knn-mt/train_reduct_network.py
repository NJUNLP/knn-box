import argparse
from knnbox.datastore.pck_datastore import PckDatastore, TripletDatastoreSamplingDataset, ReductionNetwork


if __name__ == "__main__":
    parser = argparse.ArgumentParser("train reduct network")
    parser.add_argument("--dataset-sample-rate", type=float,
                        help="the ratio to sample the dataset")
    parser.add_argument("--reduct-dim", type=int,
                        help="the dimension we want to reduct to")
    parser.add_argument("--batch-size", type=int,
                        help="the batch size to train the network")
    parser.add_argument("--learning-rate", type=float,
                        help="the learning rate")
    parser.add_argument("--min-learning-rate", type=float,
                        help="the minimum learning rate")
    parser.add_argument("--patience", type=int,
                        help="the patience to train")
    parser.add_argument("--valid-interval", type=int,
                        help="how many steps do a validation")
    parser.add_argument("--dr-loss-ratio", type=float,
                        help="the ratio of distance ranking loss in total loss")
    parser.add_argument("--nce-loss-ratio", type=float,
                        help="the ratio of the NCE loss in total loss")
    parser.add_argument("--wp-loss-ratio", type=float,
                        help="the ratio if the word prediction loss in total loss")
    parser.add_argument("--max-update", type=int,
                        help="the max update steps")
    parser.add_argument("--datastore-load-path", type=str,
                        help="the path of the datastore to load")
    parser.add_argument("--log-path", type=str,
                        help="the path to save the tensorboard log file")
    args = parser.parse_args()

    # lood the datastore
    datastore = PckDatastore.load(args.datastore_load_path, load_list=[
                                  "keys", "vals"], load_network=False)
    # create the reduction network
    datastore.reduction_network = ReductionNetwork(
        dictionary_len=datastore.dictionary_len,
        input_dim=datastore["keys"].shape[-1],
        output_dim=args.reduct_dim,
        train_mode=True,
    )
    datastore.reduction_network_input_dim = datastore["keys"].shape[-1]
    datastore.reduction_network_output_dim = args.reduct_dim

    # load the dataset
    triplet_dataset = TripletDatastoreSamplingDataset(
        dictionary_len=datastore.dictionary_len,
        use_cluster=True,
        verbose=True,
        db_keys=datastore["keys"].data,
        db_vals=datastore["vals"].data,
        sample_rate=args.dataset_sample_rate,
    )

    # train the reduction network
    datastore.train_reduction_network(
        triplet_dataset,
        args.batch_size,
        args.dr_loss_ratio,
        args.nce_loss_ratio,
        args.wp_loss_ratio,
        args.learning_rate,
        args.min_learning_rate,
        args.patience,
        args.max_update,
        args.log_path,
        args.valid_interval,
        device="cuda:0",
    )

    # dump the datastore the load path (actually dump the network)
    datastore.dump(dump_network=True)
