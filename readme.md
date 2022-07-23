--------------------------------------------------------------------------------

libds is a toolkit allows researchers and developers to easily build datastore and then embed the datastore into existing neural machine translation models or other neural network models.

This library is highly inspired by KNN-LM, adaptive KNN-MT's code.

We provide reference implementations of various papers:

* **Nearest Neighbor Machine Translation (KNN-MT)**
  + [Nearest Neighbor Machine Translation (Urvashi et al., 2021)](https://arxiv.org/abs/2010.00710)
* **Adaptive KNN-MT**
  + [ Adaptive Nearest Neighbor Machine Translation (Zheng et al., 2019)](https://arxiv.org/abs/2105.13022v1)
* **Kernel Smoothed KNN-MT**
  + [Learning Kernel-Smoothed Machine Translation with Retrieved Examples](https://arxiv.org/abs/2109.09991)

### Features:
* very small amount of code to build a datastore and apply it
* Treat various KNN systems as three modules
  - datastore
  - retriever
  - integrator

    You only need to modify the modules you care about to implement new ideas, instead of implementing the entire KNN system from scratch
* The pre implement retrievers are based on faiss-gpu, which has fast retrieve speed
* Not only KNN-MT, this library can be used to build KNN-LM and other KNN neural models.


# Getting Started

## how to install it
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

## build datastore
A typical datastore consist of keys and values. keys come from neural model's hidden state, values come from dataset.
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

## apply datastore to NMT
a typeical knn-mt consit of three modules: datastore, retriever, integrator.
```python
from libds.datastore import Datastore
from libds.retriever import Retriever
from libds.integrator import Integrator

ds = Datastore.load(path="...")
retriever = Retriever(ds, k=8)
integrator = Integrator(temperature=10, lamda_=0.7)
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
    knn_prob = integrator.get_knn_prob(**retriever.results, device=net_output[0].device)
    integrated_prob = integrator.get_integrated_prob(knn_prob, net_output[0], log_probs=log_probs)
    return integrated_prob
```

**Congralations!!** now you got a knn-mt system.
## Advanced Usage
combining different types of datastores, retrievers, Integrators, we can obtain various models. We alreay supply:
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

The complete tutorial can be found here:
1. [implement vanilla knn-mt use libds]()
2. [implement adaptive knn-mt use libds]()
3. [implement kernel smoothed knn-mt use libds]()



# Pre-trained models and examples

We provide pre-build datastores and pre-trained adaptive knn-mt model, kernel smoothed model.

* 

# License

libds is MIT-licensed.
The license applies to the pre-trained models as well.
```
