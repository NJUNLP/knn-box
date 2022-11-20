# kNN-box
[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Active](http://img.shields.io/badge/Status-Active-green.svg)](https://tterb.github.io) [![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)

kNN-box is an open-source toolkit to build kNN-MT models. We take inspiration from the code of [kNN-LM](https://github.com/urvashik/knnlm) and [adaptive kNN-MT](https://github.com/zhengxxn/adaptive-knn-mt), and develope this more extensible toolkit based on [fairseq](https://github.com/facebookresearch/fairseq). Via kNN-box, users can easily implement different kNN-MT baseline models and further develope new models.

In this toolkit, we provide reference implementations of various papers:
* [ Nearest Neighbor Machine Translation (Khandelwal et al., 2021)](https://openreview.net/pdf?id=7wCBOfJ8hJM)
* [ Adaptive Nearest Neighbor Machine Translation (Zheng et al., 2021)](https://aclanthology.org/2021.acl-short.47.pdf)
* [ Learning Kernel-Smoothed Machine Translation with Retrieved Examples (Jiang et al., 2021)](https://aclanthology.org/2021.emnlp-main.579.pdf)
* [ Efficient Machine Translation Domain Adaptation (PH Martins et al., 2022) ](https://aclanthology.org/2022.spanlp-1.3.pdf)


## Features:
* easy-to-use: a few lines of code to deploy a kNN-MT model
* research-oriented: provide implementations of various papers
* extensible: easy to develope new kNN-MT models with our toolkit.

## Requirements and Installation
* python >= 3.7
* pytorch >= 1.10.0
* faiss-gpu >= 1.7.3
<details>
<summary><b><ins>Click here before installing faiss !!!</ins></b></summary>
You should never install faiss with pip like this:

```bash
pip install faiss
```
the faiss library installed by pip will cause many problems.

You should install faiss with conda:

```bash
CPU version only:
conda install faiss-cpu -c pytorch

GPU version:
conda install faiss-gpu -c pytorch # For CUDA
```
</details>

* sacremoses == 0.0.41
* sacrebleu == 1.5.1
* fastBPE == 0.1.0

You can install this toolkit by
```shell
git clone git@github.com:NJUNLP/knn-box.git
cd knn-box
pip install --editable ./
```

## Overview
Basically, there are two steps for runing a kNN-MT model: building datastore and translating with datastore. In this toolkit, we unify different kNN-MT variants into a single framework, albeit they manipulate datastore in different ways. Specifically, the framework consists of three modules (basic class):
* **datastore**: save translation knowledge in key-values pairs
* **retriever**: retrieve useful translation knowledge from the datastore
* **combiner**: produce final prediction based on retrieval results and NMT model

Users can easily develope different kNN-MT models by customizing three modules. This toolkit also provide example implementations of various popular kNN-MT models and push-button scripts to run them, enabling researchers conveniently reproducing their experiment results.

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
<summary><b><ins>Run base neural machine translation model (our baseline)</ins></b></summary>
To translate using base neural model, execute the following command:

```bash
cd knnbox-scripts/base-nmt
bash inference.sh
```
</details>
<details>
<summary><b><ins>Run vanilla knn-mt</ins></b></summary>
To translate using knn-mt, execute the following command:

```bash
cd knnbox-scripts/vanilla-knn-mt
# step 1. build datastore
bash build_datastore.sh
# step 2. inference
bash inference.sh
```
</details>
<details>
<summary><b><ins>Run adaptive knn-mt</ins></b></summary>
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
<summary><b><ins>Run kernel smoothed knn-mt</ins></b></summary>
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
<summary><b><ins>Run greedy merge knn-mt</ins></b></summary>
implementation of [ Efficient Machine Translation Domain Adaptation (PH Martins et al., 2022) ](https://aclanthology.org/2022.spanlp-1.3.pdf)

To translate using Greedy Merge knn-mt, execute the following command:

```bash
cd knnbox-scripts/greedy-merge-knn-mt
# step 1. build datastore and prune using greedy merge method
bash build_datastore_and_prune.sh
# step 2. inferece (You can decide whether to use cache by --enable-cache)
bash inference.sh
```
</details>

![](https://s1.ax1x.com/2022/07/30/vioW4K.png)

## Visualize the kNN-MT translation process
with knnbox, you can easily obtain a web page to visualize the kNN-MT translation process interactively, and the web page will also display some useful information about the constructed datastore. 

Install the requirements:
* streamlit >= 1.13.0
* matplotlib >= 3.5.3
* seaborn >= 0.12.1
* scikit-learn >= 1.0.2

and then execute the following command:
```bash
cd knnbox-scripts/vanilla-knn-mt-visual
# step 1. build datastore for visualization 
# (the visual datastore will save addtional information compared to noraml version datastore)
bash build_datastore_visual.sh
# step 2. Configure the information for the model you want to visualize
vim model_configs.yml #Refer to the example configuration to modify this file.
# step 3. lanuch the web page
bash start_app.sh
```
You will obtain a web page look like this. Have fun playing with it.
![](https://s1.ax1x.com/2022/11/15/zVMxtx.png)
![](https://s1.ax1x.com/2022/11/15/zVQKgS.png)

<!--
## Create New KNN Models[]
If you are not only satisfied with running scripts, but want to use knnbox toolkit to create novel knn models, here are tutorials on how we implemented three example models. Hope it can help you.

* [Vanilla kNN-MT](knnbox/tutorials/markdowns/vanilla_knn_mt.md)
* [Kernel Smoothed kNN-MT](knnbox/tutorials/markdowns/kernel_smoothed_knn_mt.md)
* [Adaptive kNN-MT](knnbox/tutorials/markdowns/adaptive_knn_mt.md)
-->
## Contributor
Qianfeng Zhao: qianfeng@smail.nju.edu.cn
Wenhao Zhu: zhuwh@smail.nju.edu.cn
