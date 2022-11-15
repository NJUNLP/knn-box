 In this tutorial, we will show you how to build a knn-mt system from scratch with libds. we use WMT19 DE-EN transformer model as our base model, and multi-domain DE-EN dataset. In this tutorial, we only focus on the `Medical` domain.


## Prepare
1. download [fairseq](https://github.com/facebookresearch/fairseq/tree/v0.12.2) library and install it. when we write this tutorial, the latest version is v0.12.2, so we recommand you use the same version for convenient.
2. download libds and put the libds folder under fairseq folder.
3. download the [pretrained WMT19 DE-EN transformer model](https://github.com/facebookresearch/fairseq/blob/main/examples/wmt19/README.md).
4. download [raw dataset](https://github.com/roeeaharoni/unsupervised-domain-clusters), you should preprocess them with moses toolkits and the bpe-codes provided by pre-trained model. For Convenice, We also provide [pre-processed data](https://drive.google.com/file/d/18TXCWzoKuxWKHAaCRgddd6Ub64klrVhV/view).

now your workspace should like this:
```
    - fairseq
        - fairseq-cli
        - fairseq
        - examples
        - scripts
        - libds
```

## stage 1. build datastore
let us traverse the `Medical` training set and build a datastore.

1. add values
open **fariseq-cli/validate.py**
at the begining of the file, we declare a datastore and registe it as a global variable so that we can access it in multiple files
```python
from libds.datastore.datastore import Datastore
from libds.utils import get_registered_datastore, registe_datastore, keys_mask_select
if get_registered_datastore("ds") is None:
    ds = Datastore("/data1/zhaoqf/Retrieval-Enhanced-QE-main/libds_datastore/wmt19_medical",
        key_dim=512, value_dim=1)
    registe_datastore("ds", ds)
else:
    ds = get_registered_datastore("ds")
```

and then, when traverse dataset, we add values to datastore:
```python
  for i, sample in enumerate(progress):
        sample = utils.move_to_cuda(sample) if use_cuda else sample

        # >>>> add values to datastore.
        non_pad_tokens, mask = filter_pad_tokens(sample["target"])
        ds.add_value(non_pad_tokens)
        ds.set_mask(mask)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        _loss, _sample_size, log_output = task.valid_step(sample, model, criterion)
        progress.log(log_output, step=i)
        log_outputs.append(log_output)
```


2. add keys
open **fairseq/models/transformer.py**
at the begining of the file, we declare a datastore and registe it as a global variable so that we can access it in multiple files
```
from libds.datastore.datastore import Datastore
if get_registered_datastore("ds") is None:
    ds = Datastore("/data1/zhaoqf/Retrieval-Enhanced-QE-main/libds_datastore/wmt19_meidcal",
        key_dim=1024, value_dim=1)
    registe_datastore("ds", ds)
else:
    ds = get_registered_datastore("ds")
```

and then, in the forward function of decoder, we add keys to datastore:
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
        
        # add keys >>>>>>>>>>>>>>
        keys = keys_mask_select(x, ds.get_mask())
        ds.add_key(keys)
        # <<<<<<<<<<<<<<<<<<<<<<<<
        if not features_only:
            x = self.output_layer(x)
        return x, extra
```

3. save datastore
open **fariseq-cli/validate.py**
in the end of the file, add code to dump the datastore and faiss index to disk:
```python
if __name__ == "__main__":
    cli_main()
    # dump to disk >>>>>>>>>
    ds.dump()
    ds.build_faiss_index()
    # <<<<<<<<<<<<<<<<<<<<<
```

ok, let's run the normal validate script:
```bash
RESOURCE=/data1/zhaoqf/adaptive-knn-mt

RESOURCE_MODEL=/data1/zhaoqf/adaptive-knn-mt/wmt19.de-en/wmt19.de-en.ffn8192.pt 
SAVE_DIR=/data1/zhaoqf/trash/libds_datastore/medical
DATA_PATH=$RESOURCE/data-bin/medical
PROJECT_PATH=/data1/zhaoqf/Retrieval-Enhanced-QE-main

CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/fairseq_cli/validate.py $DATA_PATH \
--task translation \
--path $RESOURCE_MODEL \
--dataset-impl mmap \
--valid-subset train \
--model-overrides "{'eval_bleu': False, 'required_seq_len_multiple':1, 'load_alignments': False}" \
--skip-invalid-size-inputs-valid-test \
--max-tokens 1024 \
--max-tokens-valid 10000 \
--bpe fastbpe
```

you will get a datastore in `/data1/zhaoqf/Retrieval-Enhanced-QE-main/libds_datastore/wmt19_medical`.

## stage 2. Inference
apply a datastore to inference is easy too.
1. create a retriever and a combiner using the datastore you just built

open **fairseq/models/transformer.py**
at the begining of the file, create a retriever and a combiner
```python
from libds.datastore import Datastore
from libds.retriever import Retriever
from libds.combiner import Combiner
ds = Datastore.load("/data1/zhaoqf/Retrieval-Enhanced-QE-main/libds_datastore/medical")
retriever = Retriever(datastore=ds, k=32)
combiner = Combiner(lambda_=0.8, temperature=10, probability_dim=42024)
```
probability_dim is the dimension of the output probability, known as vocabulary size.

2. retrieve
open **fairseq/models/transformer.py**
in the decoder forward function, write retrieve code
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
the retrieved result will saved in retriever variable.

3. combine probability
open **fairseq/models/transformer.py**
we overwrite the `get_normalized_probs` of the transformer decoder to modify the probability caclulation process.
```python
 def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        
        knn_prob = combiner.get_knn_prob(**retriever.results, device=net_output[0].device)
        combined_prob = combiner.get_integrated_prob(knn_prob, net_output[0], log_probs=log_probs)
        return combined_prob
```

ok, we have got a knn-mt system, now we run normal inference script:
```bash
RESOURCE=/data1/zhaoqf/adaptive-knn-mt
RESOURCE_MODEL=/data1/zhaoqf/adaptive-knn-mt/wmt19.de-en/wmt19.de-en.ffn8192.pt
DATA_PATH=$RESOURCE/data-bin/medical
PROJECT_PATH=/data1/zhaoqf/Retrieval-Enhanced-QE-main

CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/fairseq_cli/generate.py $DATA_PATH \
--task translation \
--path $RESOURCE_MODEL \
--dataset-impl mmap \
--beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
--gen-subset test \
--model-overrides "{'eval_bleu': False, 'required_seq_len_multiple':1, 'load_alignments': False}" \
--skip-invalid-size-inputs-valid-test \
--max-tokens 1024 \
--max-tokens-valid 10000 \
--scoring sacrebleu \
--tokenizer moses --remove-bpe
```

the output is the knn-mt translation result.


