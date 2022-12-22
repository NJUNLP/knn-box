:<<! 
[script description]: save PLAC drop index given k_p
[dataset]: multi domain DE-EN dataset
[base model]: WMT19 DE-EN
!
# this line speed up faiss
export OMP_WAIT_POLICY=PASSIVE

PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
PLAC_DATASTORE_SAVE_PATH=$PROJECT_PATH/datastore/plac/it
PLAC_K=4
PLAC_BSZ=4096

CUDA_VISIBLE_DEVICES=2 python $PROJECT_PATH/knnbox-scripts/plac-knn-mt/save_drop_index.py \
    --plac-datastore-path $PLAC_DATASTORE_SAVE_PATH \
    --plac-k $PLAC_K \
    --plac-bsz $PLAC_BSZ
