# kernel smoothed knn-mt, train KSTER network

PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
BASE_MODEL=$PROJECT_PATH/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt
DATA_PATH=$PROJECT_PATH/data-bin/it
SAVE_DIR=$PROJECT_PATH/save-models/combiner/kernel_smooth/it

export MODE=train_kster
export DATASTORE_LOAD_PATH=$PROJECT_PATH/datastore/vanilla/it
export PROBABILITY_DIM=42024
export KEY_DIM=1024
export K=16
export COMBINER_SAVE_PATH=$SAVE_DIR

## using paper's setting
## here we specify max-update=2000 for time saving, original paper update 30000 steps.
CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/fairseq_cli/train.py $DATA_PATH \
--task translation --arch transformer_wmt19_de_en \
--train-subset train --valid-subset valid \
--best-checkpoint-metric "loss" --patience 30 --max-epoch 500 --max-update 2000 \
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
--update-freq 36


export MODE=""
