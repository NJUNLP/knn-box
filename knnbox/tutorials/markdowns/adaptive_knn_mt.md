In this tutorial, we will show you how to build a adaptive knn-mt system from scratch with knnbox. we use WMT19 DE-EN transformer model as our base model, and multi-domain DE-EN dataset. In this tutorial, we only focus on the `Medical` domain.

adaptive knn-mt is similar to vanilla knn-mt except for a different combiner. adaptive knn-mt's combiner contains a small neural network to caclulate k, temperate, lambda parameters, and we should train this network.


## Prepare
similar to vanilla knn-mt, if you haven't read that tutorial, read it. (Attention!! Step 5 is important)

Besides, before we using wmt19 model, we should define its arch. open `fairseq/models/transformer.py`, at the end of the file, add codes:

```python
# >>>>>> add by knnbox
@register_model_architecture("transformer", "transformer_wmt19_de_en")
def transformer_wmt19_de_en(args):
    args.dropout = getattr(args, "dropout", 0.2)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 8192)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    transformer_wmt_en_de_big(args)
# <<<<<<<<<<<<<<<<<<<<<<
```


## stage 1. build datastore
same to vanilla knn-mt, if you don't know how to build it, read last tutorial.


## stage 2. train meta-k network
open **fairseq/models/transformer.py**
same to vanilla knn-mt, we declare retriver and combiner first. here we use AdaptiveCombiner
```python
from knnbox.datastore import Datastore
from knnbox.retriever import Retriever
from knnbox.combiner import AdaptiveCombiner
ds = Datastore.load("/home/zhaoqf/fairseq/datastore/wmt19_medical")
retriever = Retriever(datastore=ds, k=32)
```

**attention:** because AdaptiveCombiner contains a network, it should be declared inside transformer class, otherwise we can't train it.
so in the end of TransformerDecoder class' __init__ function, we declare it.
```python
class TransformerDecoder(FairseqIncrementalDecoder):
    def __init__( ... ):
        ...
        self.combiner = AdaptiveCombiner(probability_dim=42024)
```

when we train meta-k network, we should only enable the meta-k network gridient, so in the end of Transformer __init__ fucntion:
```python
from knnbox.utils import disable_model_grad, enable_module_grad
class TransformerModel(FairseqEncoderDecoderModel):
    def __init__( ... ):
        ...
        disable_model_grad(self)
        enable_module_grad(self, "combiner")
```

modify the decoder's `forward` function to retrieve:
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
        retriever.retrieve(x)
        # <<<<<<<<<<<<<<<<<<<<<<<<
        if not features_only:
            x = self.output_layer(x)
        return x, extra
```
the retrieved result will be saved in retriever variable.

we overwrite the `get_normalized_probs` of the transformer decoder to modify the probability caclulation process.
```python
 def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        
        knn_prob = self.combiner.get_knn_prob(**retriever.results, device=net_output[0].device)
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

ok, now run the normal train script of fariseq to train meta-k:
```bash
PROJECT_PATH=/home/zhaoqf/fairseq
BASE_MODEL=$PROJECT_PATH/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt
DATA_PATH=$PROJECT_PATH/data-bin/medical

export MODE=train_metak
export COMBINER_SAVE_PATH=/home/zhaoqf/fairseq/save-models/adaptive/combiner/medical

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

## recover environment variable
export MODE=""
```

you should get the combiner in `home/zhaoqf/fairseq/save-models/adaptive/combiner/medical` after training.

## stage 3. inference
open **fairseq/models/transformer.py**
same to vanilla knn-mt but load a AdaptiveCombiner from disk.
```python
from knnbox.datastore import Datastore
from knnbox.retriever import Retriever
from knnbox.combiner import AdaptiveCombiner
ds = Datastore.load("/home/zhaoqf/fairseq/datastore/wmt19_medical")
retriever = Retriever(datastore=ds, k=32)
```
in the decoder __init__ function, load the trained combiner from disk:
class TransformerDecoder(FairseqIncrementalDecoder):
    def __init__( ... ):
        ...
        self.combiner = AdaptiveCombiner.load("home/zhaoqf/fairseq/save-models/adaptive/combiner/medical")
```

all of the other modification are same as below: `forward function`, `get_normalized_probs`.

finally, run inference script(same as vaniall knn-mt), we will get adaptive knn-mt translation result.


