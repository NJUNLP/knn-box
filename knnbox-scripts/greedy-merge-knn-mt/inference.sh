:<<! 
[script description]: use greedy merge knn-mt to translate
[dataset]: multi domain DE-EN dataset
[base model]: WMT19 DE-EN

Note 1: My test results show that using cache is faster only when the batch is small, 
for example batch-size equals 16. When the batch is large, there are more entries inside the cache
(because we clear the cache only when a new batch comes), 
and inference speed maybe slower than not adding a cache. so I recommand you
not set --enable-cache when batch is large.

Note 2. we follow the original paper to use beam size 5, lenpen 1.0, --max-len-a 0, --max-len-b 200.
!
# this line speed up faiss
export OMP_WAIT_POLICY=PASSIVE

PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
BASE_MODEL=$PROJECT_PATH/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt
DATA_PATH=$PROJECT_PATH/data-bin/medical
DATASTORE_LOAD_PATH=$PROJECT_PATH/datastore/greedy-merge/medical_pca256_merge2

CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/knnbox-scripts/common/generate.py $DATA_PATH \
--task translation \
--path $BASE_MODEL \
--dataset-impl mmap \
--beam 5 --source-lang de --target-lang en \
--gen-subset test \
--scoring sacrebleu \
--batch-size 8 \
--tokenizer moses \
--remove-bpe \
--user-dir $PROJECT_PATH/knnbox/models \
--arch greedy_merge_knn_mt@transformer_wmt19_de_en \
--knn-mode inference \
--knn-datastore-path $DATASTORE_LOAD_PATH \
--knn-k 4 \
--knn-lambda 0.8 \
--knn-temperature 10.0 \
--enable-cache --cache-threshold 6.0 \
--use-merge-weights \
