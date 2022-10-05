In this tutorial, we will show you how to build a kernel smoothed knn-mt system from scratch with libds. we use WMT19 DE-EN transformer model as our base model, and multi-domain DE-EN dataset. In this tutorial, we only focus on the `Medical` domain.

kernel smoothed knn-mt's build process is very similar to adaptive knn-mt, they both has a trainable combiner. In kernel smoothed knn-mt, the trainable network named KSTER.


## Prepare
same to vanilla knn-mt, if you haven't read that tutorial, read it.


## stage 1. build datastore
same to vanilla knn-mt, if you haven't read that tutorial, read it.


## stage 2. train KSTER network
open **fairseq/models/transformer.py**

same to vanilla knn-mt, we declare retriever and combiner first. here we use KernelSmoothedCombiner.
```python
from knnbox.datastore import Datastore
from knnbox.retriever import Retriever
from knnbox.combiner import KernelSmoothedCombiner
ds = Datastore.load("/home/zhaoqf/fairseq/datastore/wmt19_medical")
retriever = Retriever(datastore=ds, k=32)
```

**attention:** because KernelSmoothedCombiner contains a network, it should be declared inside transformer class, otherwise we can't train it.
so in the end of TransformerDecoder class' __init__ function, we declare it.
```python
class TransformerDecoder(FairseqIncrementalDecoder):
    def __init__( ... ):
        ...
        self.combiner = KernelSmoothedCombiner(query_dim=1024, probability_dim=42024)
```

when we train KSTER network, we should only enable the KSTER network gridient, so in the end of Transformer __init__ fucntion:

```python
from knnbox.utils import disable_model_grad, enable_module_grad
class TransformerModel(FairseqEncoderDecoderModel):
    def __init__( ... ):
        ...
        disable_model_grad(self)
        enable_module_grad(self, "combiner")
```

modify decoder's forward function, because the KernelSmoothedCombiner need `query` and `keys` as input, we specify `return_keys=True`and`return_query=True` paramertes when retrieve. 
```python
 def forward( ... ):

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        
        # retrieve >>>>>>>>>>>>>>
        retriever.retrieve(x, return_keys=True, return_value=True)
        # <<<<<<<<<<<<<<<<<<<<<<<<
        if not features_only:
            x = self.output_layer(x)
        return x, extra
```
the retrieved result will be saved in retriever variable.


 we overwrite the `get_normalized_probs` of the transformer decoder to modify the probability caclulation process.
 remeber to specify **train_KSTER=True** when calling get_knn_prob.(kster drop nearest key-value pair when training)
```python
 def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        
        knn_prob = self.combiner.get_knn_prob(**retriever.results, device=net_output[0].device, train_KSTER=True)
        combined_prob = self.combiner.get_combined_prob(knn_prob, net_output[0], log_probs=log_probs)
        return combined_prob
```


When training the combiner, if a best checkpoint is obtained, we only want to store the combiner module to disk instead of storing the entire model. so open `fairseq/checkpoint_utils.py` and find `save_checkpoint` function, modify the logic of saving checkpoint.

```python
checkpoints = [
        os.path.join(args.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond
    ]
    if len(checkpoints) > 0:
        ## knn-box add code >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        MODE = os.environ["MODE"]
        if MODE == "train_kster" or MODE == "train_metak":
            if checkpoint_conds["checkpoint_best{}.pt".format(suffix)]:
                COMBINER_SAVE_PATH = os.environ["COMBINER_SAVE_PATH"]
                trainer.model.decoder.combiner.dump(COMBINER_SAVE_PATH)
                logger.info("dumped combiner to {}".format(COMBINER_SAVE_PATH))
        else:
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            trainer.save_checkpoint(checkpoints[0], extra_state)
            for cp in checkpoints[1:]:
                PathManager.copy(checkpoints[0], cp, overwrite=True)

            write_timer.stop()
            logger.info(
                "saved checkpoint {} (epoch {} @ {} updates, score {}) (writing took {} seconds)".format(
                    checkpoints[0], epoch, updates, val_loss, write_timer.sum
                )
            )
```

ok, now run the normal train script of fariseq to train kster:
```bash
# kernel smoothed knn-mt, train KSTER network

PROJECT_PATH=/home/zhaoqf/fairseq
BASE_MODEL=$PROJECT_PATH/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt
DATA_PATH=$PROJECT_PATH/data-bin/it
SAVE_DIR=$PROJECT_PATH/save-models/kernel_smooth/combiner/medical

export MODE=train_kster
export DATASTORE_LOAD_PATH=$PROJECT_PATH/datastore/vanilla/medical
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
--tensorboard-logdir $SAVE_DIR/log \
--save-dir $SAVE_DIR \
--max-tokens 1024 \
--update-freq 36


export MODE=""
```
you should get the combiner in `home/zhaoqf/fairseq/save-models/kernel_smoothed/combiner/medical` after training.

## stage 3. inference
open **fairseq/models/transformer.py**
same to vanilla knn-mt but load a AdaptiveCombiner from disk.
```python
from knnbox.datastore import Datastore
from knnbox.retriever import Retriever
from knnbox.combiner import KernelSmoothedCombiner
ds = Datastore.load("/home/zhaoqf/fairseq/datastore/wmt19_medical")
retriever = Retriever(datastore=ds, k=32)
```

in the decoder __init__ function, load the trained combiner from disk:
```python
class TransformerDecoder(FairseqIncrementalDecoder):
    def __init__( ... ):
        ...
        self.combiner = KernelSmoothedCombiner.load("home/zhaoqf/fairseq/save-models/kernel_smoothed/combiner/medical")
```

decoder's forward function are same as training,  `get_normalized_probs` is a little different from training, we specify
train_KSTER=False when inference(retain nearest key-value pairs when inference):

```python
 def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        
        knn_prob = self.combiner.get_knn_prob(**retriever.results, device=net_output[0].device, train_KSTER=False)
        combined_prob = self.combiner.get_combined_prob(knn_prob, net_output[0], log_probs=log_probs)
        return combined_prob
```

finally, run inference script(same as vaniall knn-mt's script), we will get kernel smoothed knn-mt translation result.


