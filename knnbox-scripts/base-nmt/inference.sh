:<<!
[script description]: use neural machine translation model to translate 
[dataset]: multi domain DE-EN dataset
[base model]: WMT19 DE-EN
!

PROJECT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
DATA_PATH=$PROJECT_PATH/data-bin/koran
BASE_MODEL=$PROJECT_PATH/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt

CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/fairseq_cli/generate.py $DATA_PATH \
--task translation \
--path $BASE_MODEL \
--dataset-impl mmap \
--beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
--gen-subset test \
--model-overrides "{'eval_bleu': False, 'required_seq_len_multiple':1, 'load_alignments': False}" \
--max-tokens 4096 \
--scoring sacrebleu \
--tokenizer moses --remove-bpe \
