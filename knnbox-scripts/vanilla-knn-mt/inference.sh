:<<! 
[script description]: use vanilla-knn-mt to translate
[dataset]: multi domain DE-EN dataset
[base model]: WMT19 DE-EN
!

PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
BASE_MODEL=$PROJECT_PATH/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt
DATA_PATH=$PROJECT_PATH/data-bin/medical


CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/knnbox-scripts/common/generate.py $DATA_PATH \
--task translation \
--path $BASE_MODEL \
--dataset-impl mmap \
--beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
--gen-subset test \
--max-tokens 1024 \
--max-tokens-valid 10000 \
--scoring sacrebleu \
--tokenizer moses \
--remove-bpe \
--user-dir $PROJECT_PATH/knnbox/models \
--arch vanilla_knn_mt@transformer_wmt19_de_en \
--knn-mode inference \
--knn-datastore-path $PROJECT_PATH/datastore/vanilla/medical \
--knn-k 4 \
--knn-lambda 0.8 \
--knn-temperature 10.0 \



