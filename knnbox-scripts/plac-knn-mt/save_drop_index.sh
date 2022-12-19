:<<! 
[script description]: save PLAC drop index given k_p
[dataset]: multi domain DE-EN dataset
[base model]: WMT19 DE-EN
!
# this line speed up faiss
export OMP_WAIT_POLICY=PASSIVE

PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
BASE_MODEL=$PROJECT_PATH/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt
DATA_PATH=$PROJECT_PATH/data-bin/it
DATASTORE_SAVE_PATH=$PROJECT_PATH/datastore/plac/it


CUDA_VISIBLE_DEVICES=2 python $PROJECT_PATH/knnbox-scripts/plac-knn-mt/save_drop_index.py \
    --plac-datastore-path $DATASTORE_SAVE_PATH \
    --plac-k 4 \
    --plac-bsz 4096
