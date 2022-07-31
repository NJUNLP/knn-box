--------------------------------------------------------------------------------

libds is a toolkit allows researchers and developers to **easily** build datastore and embed the datastore into existing neural machine translation models or other neural network models.

This library is highly inspired by [KNN-LM](https://github.com/urvashik/knnlm), [adaptive KNN-MT](https://github.com/zhengxxn/adaptive-knn-mt)'s code.

We provide reference implementations of various papers:

* **Nearest Neighbor Machine Translation (KNN-MT)**
  + [Nearest Neighbor Machine Translation (Urvashi et al., 2021)](https://arxiv.org/abs/2010.00710)
* **Adaptive KNN-MT**
  + [ Adaptive Nearest Neighbor Machine Translation (Zheng et al., 2019)](https://arxiv.org/abs/2105.13022v1)
* **Kernel Smoothed KNN-MT**
  + [Learning Kernel-Smoothed Machine Translation with Retrieved Examples](https://arxiv.org/abs/2109.09991

# Features:
* Very small amount of code to build a datastore and apply it
* Easy to modify some compoents and implment new idea
* Treat various KNN systems as three modules
  - datastore: build or load datastore
  - retriever: retrieve key value pairs from datastore with query
  - combiner: caclulate knn probability and combine knn probability with neural model probability

    You only need to modify individual module you care about to implement new ideas, instead of implementing the entire KNN system from scratch
* The pre implement retrievers are based on faiss-gpu, which has fast retrieve speed
* Not only KNN-MT, this library can be used to build KNN-LM and other KNN neural models


# Getting Started
## !!Attention
If you don't want to know how to use libds library but **just want to use our pre-implemeted knn models**(vanilla knn-mt, adaptive knn-mt, etc.).
use [libds-exmaples](https://github.com/ZhaoQianfeng/libds-examples) instead, follow it's instrcutions and it will save your time.

If you still want to know how libds works and how to implement your own knn models using libds, Please read on.
## how to install libds
Just copy this libds folder to your project folder.

For example, if you are using fairseq framework, the expected file directory is like this:
```
    - fairseq
        - fairseq-cli
        - fairseq
        - examples
        - scripts
        - libds
        - ....
```

## workflow
![](https://s1.ax1x.com/2022/07/30/vioW4K.png)
It usually takes two stages to build KNN system on the pre-trained neural network model using libds. 

In the first stage, we traverse the training set, use the hidden layer state of the neural network model and the words of the reference translation to form key and value pairs, and add them to the datastore. 


In the second stage, we will use the constructed datastore to enhance the neural network model. we send the hidden layer state (also known as key) of the neural network model to a module called retriever.The retriever contains the datastore and uses the faiss library for retrieval. The results of the retriever module and the output probability of the neural network model are sent to a module called combiner, which is responsible for calculating the KNN probability with the retrieval distance, and then combining the KNN probability and the neural network probability to obtain the final probability.

Let's see how to write the code of these two stages.
## stage 1. Build datastore
A typical datastore consist of keys and values. keys come from neural model's hidden layer state, values come from dataset.
- initialize a new datastore
```python
from libds.datastore import Datastore
ds = Datastore(path="...")
```
- add keys to datastores
```python
ds.add_keys(keys)
```
- add values
```python
ds.add_values(values)
```
- dump the datastore to disk
```python
ds.dump()
```
- load a existing datastore from disk
```python
ds = Datastore.load(path="...")
```

## stage 2. Inference
a typeical knn-mt system consit of three modules: datastore, retriever, combiner.
```python
from libds.datastore import Datastore
from libds.retriever import Retriever
from libds.combiner import Combiner

ds = Datastore.load(path="...")
retriever = Retriever(ds, k=8)
combiner = Combiner(temperature=10, lamda_=0.7)
```
we retrieve the datastore in model's forward function. Take fairseq for example, we add one line code in the end of models' `forward` function:
```python
retriever.retrieve(x)
```
we then modify the probability caclulation function to combine knn probability with nerual probability. Take fairseq for example, we modify the `get_normalized_probs` function:
```python
def get_normalized_probs(
    self,
    net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
    log_probs: bool,
    sample: Optional[Dict[str, Tensor]] = None,
):
    knn_prob = combiner.get_knn_prob(**retriever.results, device=net_output[0].device)
    combined_prob = combiner.get_combined_prob(knn_prob, net_output[0], log_probs=log_probs)
    return combined_prob
```
**Congralations!!** now you got a knn-mt system.

## complete tutorials
1. [implement vanilla knn-mt using libds](tutorials/markdowns/vanilla_knn_mt.md)
2. [implement adaptive knn-mt using libds](tutorials/markdowns/adaptive_knn_mt.md)
3. [implement kernel smoothed knn-mt using libds](tutorials/markdowns/kernel_smoothed_knn_mt.md)

## Advanced Usage
composing different types of datastores, retrievers, Integrators, you can obtain various knn models. We alreay supply:
- datastore:
    - Datastore
    - Fast Datastore
- Retriever:
    - Retriever
    - Fast Retriever
- Integrator:
    - Integrator
    - AdaptiveIntegrator
    - KernelSmoothedIntegrator

You can  implement new datastore/retriever/integrator to build your novel knn model.




# Pre-trained models and examples

We provide pre-build datastores and pre-trained adaptive knn-mt model, kernel smoothed models.

* TODO

# License

libds is MIT-licensed.
The license applies to the pre-trained models as well.
