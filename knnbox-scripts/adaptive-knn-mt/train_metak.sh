## adaptive knn-mt, train meta-k network
PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
BASE_MODEL=$PROJECT_PATH/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt
DATA_PATH=$PROJECT_PATH/data-bin/it
SAVE_DIR=$PROJECT_PATH/save-models/combiner/adaptive/it

export MODE=train_metak
export DATASTORE_LOAD_PATH=$PROJECT_PATH/datastore/vanilla/it
export PROBABILITY_DIM=42024
export KEY_DIM=1024
export COMBINER_SAVE_PATH=$SAVE_DIR
export K=16

# using paper's settings
CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/fairseq_cli/train.py $DATA_PATH \
--task translation --arch transformer_wmt19_de_en \
--train-subset valid --valid-subset valid \
--best-checkpoint-metric "loss" \
--finetune-from-model $BASE_MODEL \
--optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
--lr 3e-4 --lr-scheduler reduce_lr_on_plateau \
--min-lr 3e-05 --criterion label_smoothed_cross_entropy --label-smoothing 0.001 \
--lr-patience 5 --lr-shrink 0.5 --patience 30 --max-epoch 500 --max-update 5000 \
--criterion label_smoothed_cross_entropy \
--save-interval-updates 100 \
--no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state \
--tensorboard-logdir $SAVE_DIR/log \
--save-dir $SAVE_DIR \
--batch-size 4 \
--update-freq 8

## --keep-best-checkpoints 1 \
## recover environment variable
export MODE=""
