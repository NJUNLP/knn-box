:<<!
prune datastore using the method of PCK-MT
1. if you need prune datastore size, use --need-prune-size
2. if you need train the dimension-reduction network, use --need-train-network
3. if you need reduct the datastore dimension, use --need-reduct-dimension
You can use these three options togther or some of them.
!

export OMP_WAIT_POLICY=PASSIVE
PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
DATASTORE_LOAD_PATH=$PROJECT_PATH/datastore/pck/koran
PRUNE_STYLE=prune_similar_ppl # when prune datastore size, use what policy
lr=3e-4 # the learning rate to train reduction network
N_OF_4_GRAM=4 # use how many n-gram, choose from 1~4
DR_LOSS_RATIO=0.0 # ratio of dr loss in total loss
NCE_LOSS_RATIO=0.9 # ratio of nce loss in total loss
WP_LOSS_RATIO=0.1 # ratio of wp loss in total loss
TRAIN_BATCH_SIZE=1000 # adjust this value according to your GPU memory
EPOCH=400   # train how many epochs


CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/knnbox-scripts/pck-knn-mt/prune_datastore.py \
--datastore-path $DATASTORE_LOAD_PATH \
--need-prune-size \
--need-train-network \
--need-reduct-dimension \
--prune-style $PRUNE_STYLE \
--n-of-4-gram $N_OF_4_GRAM \
--epoch $EPOCH \
--train-batch-size $TRAIN_BATCH_SIZE \
--dr-loss-ratio $DR_LOSS_RATIO \
--nce-loss-ratio $NCE_LOSS_RATIO \
--wp-loss-ratio $WP_LOSS_RATIO \
