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
from libds.datastore import Datastore
from libds.retriever import Retriever
from libds.combiner import KernelSmoothedCombiner
ds = Datastore.load("/data1/zhaoqf/Retrieval-Enhanced-QE-main/libds_datastore/medical")
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
from libds.utils import disable_model_grad, enable_module_grad
class TransformerModel(FairseqEncoderDecoderModel):
    def __init__( ... ):
        ...
        disable_model_grad(self)
        enable_module_grad(self, "combiner")
```

before our program exit, we need dump the trained combiner to disk, so we add a __del__ function to TransformerMdoel:
```python
class TransformerModel():
    def __del__(self):
        self.decoder.combiner.dump("/data1/zhaoqf/Retrieval-Enhanced-QE-main/libds_combiner/medical")
```

ok, now run the normal train script of fariseq to train kster:
```bash
RESOURCE=/data1/zhaoqf/adaptive-knn-mt
RESOURCE_MODEL=/data1/zhaoqf/adaptive-knn-mt/wmt19.de-en/wmt19.de-en.ffn8192.pt
SAVE_DIR=/data1/zhaoqf/Retrieval-Enhanced-QE-main/libds_combiner/medical
DATA_PATH=$RESOURCE/data-bin/medical
PROJECT_PATH=/data1/zhaoqf/Retrieval-Enhanced-QE-main

# using the paper settings 
CUDA_VISIBLE_DEVICES=7 python $PROJECT_PATH/fairseq_cli/train.py $DATA_PATH \
--task translation --arch transformer_wmt19_de_en \
--train-subset valid \
--max-epoch 80 \
--finetune-from-model $RESOURCE_MODEL \
--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
--lr 3e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--no-epoch-checkpoints \
--keep-best-checkpoints 1 \
--tensorboard-logdir $SAVE_DIR/log \
--batch-size 4 \
--update-freq 8 \
--disable-validation \
--save-dir $SAVE_DIR
```

## stage 3. inference
open **fairseq/models/transformer.py**
same to vanilla knn-mt but load a AdaptiveCombiner from disk.
```python
from libds.datastore import Datastore
from libds.retriever import Retriever
from libds.combiner import AdaptiveCombiner
ds = Datastore.load("/data1/zhaoqf/Retrieval-Enhanced-QE-main/libds_datastore/medical")
retriever = Retriever(datastore=ds, k=32)
```python
in the decoder __init__ function, load the trained combiner from disk:
```
class TransformerDecoder(FairseqIncrementalDecoder):
    def __init__( ... ):
        ...
        self.combiner = KernelSmoothedCombiner.load("/data1/zhaoqf/Retrieval-Enhanced-QE-main/libds_combiner/medical")
```

because the KernelSmoothedCombiner need `query` and `keys` as input, we specify `return_keys=True`and`return_query=True` paramertes when retrieve.
```python
        retriever.retrieve(x, return_keys=True, return_query=True)
```

all of the other modification are same as vanilla knn-mt.

finally, run inference script(same as vaniall knn-mt's script), we will get kernel smoothed knn-mt translation result.


