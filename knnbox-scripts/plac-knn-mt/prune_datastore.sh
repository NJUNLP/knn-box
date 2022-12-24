:<<! 
[script description]: save PLAC drop index given k_p
[dataset]: multi domain DE-EN dataset
[base model]: WMT19 DE-EN
!
# this line speed up faiss
export OMP_WAIT_POLICY=PASSIVE

PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
PLAC_DATASTORE_SAVE_PATH=$PROJECT_PATH/datastore/plac/it
DATASTORE_SAVE_PATH=$PROJECT_PATH/datastore/vanilla/it
PLAC_K=4
PLAC_BSZ=4096
PLAC_RATIO=0.40

CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/knnbox-scripts/plac-knn-mt/prune_datastore.py \
    --plac-datastore-path $PLAC_DATASTORE_SAVE_PATH \
    --datastore-path $DATASTORE_SAVE_PATH \
    --pruned-datastore-path ${DATASTORE_SAVE_PATH}_plac_k${PLAC_K}_ratio${PLAC_RATIO} \
    --plac-k $PLAC_K \
    --plac-ratio $PLAC_RATIO \
    --plac-bsz $PLAC_BSZ
