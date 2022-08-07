# KNN-box
[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Active](http://img.shields.io/badge/Status-Active-green.svg)](https://tterb.github.io) [![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)

kNN-box is an open-source toolkit to build kNN-MT models. We take inspiration from the code of [kNN-LM](https://github.com/urvashik/knnlm), [adaptive kNN-MT](https://github.com/zhengxxn/adaptive-knn-mt), and develope this more extensible toolkit. Via kNN-box, users can easily implement different kNN-MT baseline models and further devise new models.

In this toolkit, we provide reference implementations of various papers:
* [ Nearest Neighbor Machine Translation (Khandelwal et al., 2021)](https://openreview.net/pdf?id=7wCBOfJ8hJM)
* [ Adaptive Nearest Neighbor Machine Translation (Zheng et al., 2021)](https://aclanthology.org/2021.acl-short.47.pdf)
* [ Learning Kernel-Smoothed Machine Translation with Retrieved Examples (Jiang et al., 2021)](https://aclanthology.org/2021.emnlp-main.579.pdf)


## Features:
* easy-to-use: a few lines of code to deploy a kNN-MT model
* research-oriented: provide implementations of various papers
* extensible: easy to develope new kNN-MT models with our toolkit.

## Installation
```shell
cp kNN-box path-to-your-project/
```

## Overview
kNN-box provides three modules for implementing kNN-MT models:
* datastore: save translation knowledge in key-values pairs
* retriever: retrieve useful translation knowledge from the datastore
* combiner: produce final prediction based on retrieval results and NMT model

To illustrate the usage of kNN-box, we provide example implementations of following kNN-MT variants:
* [Vanilla kNN-MT](tutorials/markdowns/vanilla_knn_mt.md)
* [Kernel Smoothed kNN-MT](tutorials/markdowns/kernel_smoothed_knn_mt.md)
* [Adaptive kNN-MT](tutorials/markdowns/adaptive_knn_mt.md)

![](https://s1.ax1x.com/2022/07/30/vioW4K.png)


## Contributor
Qianfeng Zhao: qianfeng@smail.nju.edu.cn
Wenhao Zhu: zhuwh@smail.nju.edu.cn