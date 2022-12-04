:<<!
[script description]: train adptive-knn-mt's meta-k network
[dataset]: multi domain DE-EN dataset
[base model]: WMT19 DE-ENscript

note 1. original paper update 30k steps, but for time saving we update 5k steps here,
the result of 5k version is good enough.

note 2. You can adjust --max-tokens and --update-freq based on your GPU memory.
original paper recommand that max-tokens*update-freq equals 36864.
!
# this line speed up faiss
export OMP_WAIT_POLICY=PASSIVE

PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
BASE_MODEL=$PROJECT_PATH/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt
DATA_PATH=$PROJECT_PATH/data-bin/medical
DATASTORE_LOAD_PATH=$PROJECT_PATH/datastore/vanilla/medical
SAVE_DIR=$PROJECT_PATH/save-models/combiner/kernel_smooth/medical

## using paper's setting
CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/knnbox-scripts/common/train.py $DATA_PATH \
--task translation \
--train-subset train --valid-subset valid \
--best-checkpoint-metric "loss" --patience 30 --max-epoch 500 --max-update 5000 \
--finetune-from-model $BASE_MODEL \
--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
--lr 2e-4 --lr-scheduler inverse_sqrt --warmup-updates 200 \
--criterion label_smoothed_cross_entropy --label-smoothing 0.001 \
--save-interval-updates 100 \
--no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state \
--keep-best-checkpoints 1 \
--tensorboard-logdir $SAVE_DIR/log \
--save-dir $SAVE_DIR \
--max-tokens 1024 \
--update-freq 36 \
--arch kernel_smoothed_knn_mt@transformer_wmt19_de_en \
--user-dir $PROJECT_PATH/knnbox/models \
--knn-mode train_kster \
--knn-datastore-path $DATASTORE_LOAD_PATH \
--knn-k 16 \
--knn-combiner-path $SAVE_DIR \
