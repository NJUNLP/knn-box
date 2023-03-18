:<<! 
[script description]: use simple-scalable-knn-mt to inference
[dataset]: multi domain DE-EN dataset
[base model]: WMT19 DE-EN
!
# this line will speed up faiss
export OMP_WAIT_POLICY=PASSIVE

PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
BASE_MODEL=$PROJECT_PATH/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt
DATA_PATH=$PROJECT_PATH/data-bin/it
ELASTIC_INDEX_NAME=it

CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/knnbox-scripts/common/generate.py $DATA_PATH \
--task translation \
--path $BASE_MODEL \
--dataset-impl mmap \
--beam 4 --lenpen 0.6  --source-lang de --target-lang en \
--gen-subset test \
--max-tokens 2048 \
--scoring sacrebleu \
--tokenizer moses \
--remove-bpe \
--user-dir $PROJECT_PATH/knnbox/models \
--arch simple_scalable_knn_mt@transformer_wmt19_de_en \
--knn-mode inference \
--elastic-index-name $ELASTIC_INDEX_NAME \
--knn-k 2 \
--knn-temperature 100.0 \
--reserve-top-m 16 \



