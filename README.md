# :card_file_box: kNN-box
[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Active](http://img.shields.io/badge/Status-Active-green.svg)](https://tterb.github.io) [![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)

kNN-box is an open-source toolkit to build kNN-MT models. We take inspiration from the code of [kNN-LM](https://github.com/urvashik/knnlm) and [adaptive kNN-MT](https://github.com/zhengxxn/adaptive-knn-mt), and develope this more extensible toolkit based on [fairseq](https://github.com/facebookresearch/fairseq). Via kNN-box, users can easily implement different kNN-MT baseline models and further develope new models.

- [:card\_file\_box: kNN-box](#card_file_box-knn-box)
  - [Features](#features)
  - [Citation](#citation)
  - [Requirements and Installation](#requirements-and-installation)
  - [Overview](#overview)
  - [Visualization](#visualization)
  - [Benchmark](#benchmark)
  - [Contributor](#contributor)


## Features
* :dart: easy-to-use: a few lines of code to deploy a kNN-MT model
* :telescope: research-oriented: provide implementations of various papers
* :building_construction: extensible: easy to develope new kNN-MT models with our toolkit.
* :bar_chart: visualized: the whole translation process of the kNN-MT can be visualized

## Requirements and Installation
* python >= 3.7
* pytorch >= 1.10.0
* faiss-gpu >= 1.7.3
* sacremoses == 0.0.41
* sacrebleu == 1.5.1
* fastBPE == 0.1.0
* streamlit >= 1.13.0
* scikit-learn >= 1.0.2
* seaborn >= 0.12.1

You can install this toolkit by
```shell
git clone git@github.com:NJUNLP/knn-box.git
cd knn-box
pip install --editable ./
```

Note: Installing faiss with pip is not suggested. For stability, we recommand you to install faiss with conda

```bash
CPU version only:
conda install faiss-cpu -c pytorch

GPU version:
conda install faiss-gpu -c pytorch # For CUDA
```

## Overview
Basically, there are two steps for runing a kNN-MT model: building datastore and translating with datastore. In this toolkit, we unify different kNN-MT variants into a single framework, albeit they manipulate datastore in different ways. Specifically, the framework consists of three modules (basic class):
* **datastore**: save translation knowledge as key-values pairs
* **retriever**: retrieve useful translation knowledge from the datastore
* **combiner**: produce final prediction based on retrieval results and NMT model

<img src="https://s1.ax1x.com/2023/02/25/ppppWtK.png" width="38%" alt="design">
Users can easily develope different kNN-MT models by customizing three modules. This toolkit also provide example implementations of various popular kNN-MT models (listed below) and push-button scripts to run them, enabling researchers conveniently reproducing their experiment results:

<!-- In this toolkit, we provide reference implementations of various papers: -->
<!-- * [ What Knowledge Is Needed? Towards Explainable Memory for kNN-MT Domain Adaptation (Zhu et al., arXiv'2022)](https://arxiv.org/pdf/2211.04052v2.pdf) -->
<details>
<summary><b><ins>Preparation: download pretrained models and dataset</ins></b></summary>

You can prepare pretrained models and dataset by executing the following command:

```bash
cd knnbox-scripts
bash prepare_dataset_and_model.sh
```

> use bash instead of sh. If you still have problem running the script, you can manually download the [wmt19 de-en single model](https://github.com/facebookresearch/fairseq/blob/main/examples/wmt19/README.md) and [multi-domain de-en dataset](https://drive.google.com/file/d/18TXCWzoKuxWKHAaCRgddd6Ub64klrVhV/view), and put them into correct directory (you can refer to the path in the script).
</details>
<details>
<summary><b><ins>base neural machine translation model (our baseline)</ins></b></summary>
To translate using base neural model, execute the following command:

```bash
cd knnbox-scripts/base-nmt
bash inference.sh
```
</details>
<details>
<summary><b><ins>Nearest Neighbor Machine Translation</ins></b></summary>

Implementation of [ Nearest Neighbor Machine Translation (Khandelwal et al., ICLR'2021)](https://openreview.net/pdf?id=7wCBOfJ8hJM) 

To translate using vanilla knn-mt, execute the following command:

```bash
cd knnbox-scripts/vanilla-knn-mt
# step 1. build datastore
bash build_datastore.sh
# step 2. inference
bash inference.sh
```
</details>
<details>
<summary><b><ins>Adaptive Nearest Neighbor Machine Translation</ins></b></summary>

Implementation of [ Adaptive Nearest Neighbor Machine Translation (Zheng et al., ACL'2021)](https://aclanthology.org/2021.acl-short.47.pdf)

To translate using adaptive knn-mt, execute the following command:

```bash
cd knnbox-scripts/adaptive-knn-mt
# step 1. build datastore
bash build_datastore.sh
# step 2. train meta-k network
bash train_metak.sh
# step 3. inference
bash inference.sh
```
</details>
<details>
<summary><b><ins>Learning Kernel-Smoothed Machine Translation with Retrieved Examples</ins></b></summary>

Implementation of [ Learning Kernel-Smoothed Machine Translation with Retrieved Examples (Jiang et al., EMNLP'2021)](https://aclanthology.org/2021.emnlp-main.579.pdf)

To translate using kernel smoothed knn-mt, execute the following command:

```bash
cd knnbox-scripts/kernel-smoothed-knn-mt
# step 1. build datastore
bash build_datastore.sh
# step 2. train kster network
bash train_kster.sh
# step 3. inferece
bash inference.sh
```
</details>
<details>
<summary><b><ins>Efficient Machine Translation Domain Adaptation</ins></b></summary>

Implementation of [ Efficient Machine Translation Domain Adaptation (PH Martins et al., 2022) ](https://aclanthology.org/2022.spanlp-1.3.pdf)

To translate using Greedy Merge knn-mt, execute the following command:

```bash
cd knnbox-scripts/greedy-merge-knn-mt
# step 1. build datastore and prune using greedy merge method
bash build_datastore_and_prune.sh
# step 2. inferece (You can decide whether to use cache by --enable-cache)
bash inference.sh
```
</details>
<details>
<summary><b><ins>Efficient Cluster-Based k-Nearest-Neighbor Machine Translation</ins></b></summary>

Implementation of [ Efficient Cluster-Based k-Nearest-Neighbor Machine Translation (Wang et al., 2022)](https://aclanthology.org/2022.acl-long.154.pdf)

To translate using pck knn-mt, execute the following command:
```bash
cd knnbox-scripts/pck-knn-mt
# step 1. build datastore 
bash build_datastore.sh
# step 2. train reduction network
bash train_reduct_network.sh
# step 3. reduct datastore's key dimension using trained network
bash reduct_datastore_dim.sh
# step 4. train meta-k network
bash train_metak.sh
# step 5. inference
bash inference.sh
```

[optional] In addition to reducing dimensions, you can use the method in the paper to reduce the number of entries in the datastore.
```bash
(after step 1.)
bash prune_datastore_size.sh
```

</details>
<details>
<summary><b><ins>Towards Robust k-Nearest-Neighbor Machine Translation</ins></b></summary>

Implementation of [ Towards Robust k-Nearest-Neighbor Machine Translation (Jiang et al., EMNLP'2022)](https://arxiv.org/pdf/2210.08808.pdf)

To translate using robust knn-mt, execute the following command:

```bash
cd knnbox-scripts/robust-knn-mt
# step 1. build datastore
bash build_datastore.sh
# step 2. train meta-k network
bash train_metak.sh
# step 3. inference
bash inference.sh
```
</details>
<details>
<summary><b><ins>What Knowledge Is Needed? Towards Explainable Memory for kNN-MT Domain Adaptation</ins></b></summary>

Implementation of [ What Knowledge Is Needed? Towards Explainable Memory for kNN-MT Domain Adaptation (Zhu et al., 2022)](https://arxiv.org/pdf/2211.04052.pdf)

PLAC is a datastore pruning method based on MT-model's knowledge. To prune a full datastore (vanilla or dimension-reduced), execute the following command:

```bash
cd knnbox-scripts/plac-knn-mt
# step 1. save MT-model predictions
bash save_mt_pred.sh
# step 2. save prunable indexes
bash save_drop_index.sh
# step 3. prune a full datastore and save the pruned datastore
bash prune_datastore.sh
```
</details>

<details>
<summary><b><ins>Simple and Scalable Nearest Neighbor Machine Translation</ins></b></summary>

Implementation of [Simple and Scalable Nearest Neighbor Machine Translation](https://openreview.net/forum?id=uu1GBD9SlLe)

To translate using sk-mt, excute the following command:

```bash
cd knnbox-scripts/simple-scalable-knn-mt
# step 1. download elastic search
bash download_elasticsearch.sh
# step 2. start elastic search service on port 9200
./elasticsearch-8.6.1/bin/elasticsearch
# step 3. create elasticsearch index for corpus
bash create_elasticsearch_index.sh
# step 4. inference
bash inference.sh
```

If there is an elasticsearch-related error when executing the script, you may need to open 
`./elaticsearch-8.6.1/config/elasticsearch.yml` and disable the security features:
```
xpack.security.enabled: false
```
</details>

<img src="https://s1.ax1x.com/2023/03/01/ppi1Yse.png" width="100%" alt="benchmark">



## Visualization
With kNN-box, you can even visualize the whole translation process of your kNN-MT model.
You can launch the visualization service by running the following commands. Have fun with it!
<!--with knnbox, you can easily obtain a web page to visualize the kNN-MT translation process interactively, and the web page will also display some useful information about the constructed datastore.-->

```bash
cd knnbox-scripts/vanilla-knn-mt-visual
# step 1. build datastore for visualization (save more information for visualization)
bash build_datastore_visual.sh
# step 2. configure the model that you are going to visualize
vim model_configs.yml 
# step 3. launch the web page
bash start_app.sh

# Optional: regist your own tokenize handler function in src/tokenizer.py 
# and then use it as `--tokenizer` in model_configs.yml if necessary
```

![](https://s1.ax1x.com/2022/11/25/zY3djf.png)


## Citation
We now have a [paper](https://arxiv.org/abs/2302.13574) you can cite for the :card_file_box: knn-box toolkit:
```bibtex
@misc{zhu2023knnbox,
      title={kNN-BOX: A Unified Framework for Nearest Neighbor Generation}, 
      author={Wenhao Zhu and Qianfeng Zhao and Yunzhe Lv and Shujian Huang and Siheng Zhao and Sizhe Liu and Jiajun Chen},
      year={2023},
      eprint={2302.13574},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
