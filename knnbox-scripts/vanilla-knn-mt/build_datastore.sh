## build datastore for vanilla knn mt.
## dataset: multi domain DE-EN dataset
## base model: WMT19 DE-EN

PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
BASE_MODEL=$PROJECT_PATH/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt
DATA_PATH=$PROJECT_PATH/data-bin/it

export MODE=build_datastore
export DATASTORE_SAVE_PATH=$PROJECT_PATH/datastore/vanilla/it
export KEY_DIM=1024

CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/fairseq_cli/validate.py $DATA_PATH \
--task translation \
--path $BASE_MODEL \
--dataset-impl mmap \
--valid-subset train \
--model-overrides "{'eval_bleu': False, 'required_seq_len_multiple':1, 'load_alignments': False}" \
--skip-invalid-size-inputs-valid-test \
--max-tokens 1024 \
--max-tokens-valid 10000 \
--bpe fastbpe

# recover environment variable
export MODE=""
