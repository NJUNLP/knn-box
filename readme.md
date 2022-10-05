# kNN-box
[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Active](http://img.shields.io/badge/Status-Active-green.svg)](https://tterb.github.io) [![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)

kNN-box is an open-source toolkit to build kNN-MT models. We take inspiration from the code of [kNN-LM](https://github.com/urvashik/knnlm) and [adaptive kNN-MT](https://github.com/zhengxxn/adaptive-knn-mt), and develope this more extensible toolkit based on [fairseq](https://github.com/facebookresearch/fairseq). Via kNN-box, users can easily implement different kNN-MT baseline models and further develope new models.

In this toolkit, we provide reference implementations of various papers:
* [ Nearest Neighbor Machine Translation (Khandelwal et al., 2021)](https://openreview.net/pdf?id=7wCBOfJ8hJM)
* [ Adaptive Nearest Neighbor Machine Translation (Zheng et al., 2021)](https://aclanthology.org/2021.acl-short.47.pdf)
* [ Learning Kernel-Smoothed Machine Translation with Retrieved Examples (Jiang et al., 2021)](https://aclanthology.org/2021.emnlp-main.579.pdf)


## Features:
* easy-to-use: a few lines of code to deploy a kNN-MT model
* research-oriented: provide implementations of various papers
* extensible: easy to develope new kNN-MT models with our toolkit.

## Requirements and Installation
* pytorch version >= 1.5.0
* python version >= 3.6
* faiss-gpu >= 1.6.5
* sacremoses == 0.0.41
* sacrebleu == 1.5.1
* pytorch_scatter = 2.0.5
* 1.19.0 <= numpy < 1.20.0
You can install this toolkit by
```shell
pip install --editable ./
```

## Overview
Basically, there are two steps for runing a $k$NN-MT model: building datastore and translating with datastore. In this toolkit, we unify different $k$NN-MT variants into a single framework, albeit they manipulate datastore in different ways. Specifically, the framework consists of three modules (basic class):
* **datastore**: save translation knowledge in key-values pairs
* **retriever**: retrieve useful translation knowledge from the datastore
* **combiner**: produce final prediction based on retrieval results and NMT model

Users can easily develope different kNN-MT models by customizing three modules. This toolkit also provide example implementations of various popular $k$NN-MT models and push-button scripts to run them, enabling researchers conveniently reproducing their experiment results.
* [Vanilla kNN-MT](knnbox/tutorials/markdowns/vanilla_knn_mt.md)
* [Kernel Smoothed kNN-MT](knnbox/tutorials/markdowns/kernel_smoothed_knn_mt.md)
* [Adaptive kNN-MT](knnbox/tutorials/markdowns/adaptive_knn_mt.md)
![](https://s1.ax1x.com/2022/07/30/vioW4K.png)


## Contributor
Qianfeng Zhao: qianfeng@smail.nju.edu.cn
Wenhao Zhu: zhuwh@smail.nju.edu.cn