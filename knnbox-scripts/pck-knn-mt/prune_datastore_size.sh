:<<!
prune the datastore size using the method in paper.
for example a datastore shape: [100000, 1024]
using --prune-ratio 0.3, the output datastore size is about [30000, 1024]
!

export OMP_WAIT_POLICY=PASSIVE
PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
DATASTORE_LOAD_PATH=$PROJECT_PATH/datastore/pck/it  # the datastore which need 
OUTPUT_DATASTORE_PATH=$PROJECT_PATH/datastore/pck/it_prune0.5 # the directory to save the prune datastore

PRUNE_STYLE=prune_similar_ppl   # the prune policy
PRUNE_RATIO=0.6     # retain how much entires
N_OF_4_GRAM=4   # you can use a smaller n-gram than 4
MIN_SAMPLE_THRESHOLD=2    # the minimum size of every cluster to prune
THREAD_NUM=20   # use how many threads


CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/knnbox-scripts/pck-knn-mt/prune_datastore_size.py \
--prune-style $PRUNE_STYLE \
--prune-ratio $PRUNE_RATIO \
--n-of-4-gram $N_OF_4_GRAM \
--min-sample-threshold $MIN_SAMPLE_THRESHOLD \
--thread-num $THREAD_NUM \
--input-datastore-path $DATASTORE_LOAD_PATH \
--output-datastore-path $OUTPUT_DATASTORE_PATH \




