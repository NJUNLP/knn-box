:<<! 
[script description]: use greedy merge knn-mt to translate
[dataset]: multi domain DE-EN dataset
[base model]: WMT19 DE-EN

Note: My test results show that using cache is faster only when the batch is small, 
for example batch-size equals 16. When the batch is large, there are more entries inside the cache
(because we clear the cache only when a new batch comes), 
and inference speed maybe slower than not adding a cache. so I recommand you
not set --enable-cache when batch is large.
!

PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
BASE_MODEL=$PROJECT_PATH/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt
DATA_PATH=$PROJECT_PATH/data-bin/it


CUDA_VISIBLE_DEVICES=1 python $PROJECT_PATH/knnbox-scripts/common/generate.py $DATA_PATH \
--task translation \
--path $BASE_MODEL \
--dataset-impl mmap \
--beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
--gen-subset test \
--scoring sacrebleu \
--max-tokens 4096 \
--tokenizer moses \
--remove-bpe \
--user-dir $PROJECT_PATH/knnbox/models \
--arch greedy_merge_knn_mt@transformer_wmt19_de_en \
--knn-mode inference \
--knn-datastore-path $PROJECT_PATH/datastore/greedy-merge/it_pca256_merge2 \
--knn-k 8 \
--knn-lambda 0.7 \
--knn-temperature 10.0 \
--use-merge-weights \
--enable-cache \
