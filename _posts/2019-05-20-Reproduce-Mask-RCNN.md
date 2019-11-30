---
layout:     post
title:      Reproduce - Mask-RCNN
subtitle:   论文复现
date:       2019-05-20
author:     vhpg
header-img: img/placeholder_img.png
catalog: true
tags:
    - Deep Learning
---

#### Introduce
本文记录配置[Mask RCNN](https://github.com/matterport/Mask_RCNN)环境过程中遇到的一些问题。
由于某些原因，自己在这上面浪费了比较多的时间，这里进行记录。

#### Config
[Mask RCNN](https://github.com/matterport/Mask_RCNN)网页中已经给了环境配置说明：
![Screenshot from 2019-05-20 20-10-02](/assets/Screenshot%20from%202019-05-20%2020-10-02.png)

其中，`requirements.txt`中的tensorflow是使用pip安装，且安装的是CPU的版本，所以自己首先考虑的是使用conda安装GPU版本的tensorflow，这样相应的CUDA和cudnn环境就可自动安装好。
但是这样出现一个问题，conda中能安装的cudnn版本最高为7.3.1，而该Mask RCNN程序使用的cudnn至少要为7.4.1，并且conda中无法更新更高版本的cudnn，因此使用conda安装tensorflow这条路走不通。

使用pip也可以安装GPU版本的tensorflow，但不能在相应的conda环境中自动安装CUDA和cudnn，需要在Ubuntu本地安装，此处略过在Ubuntu上安装CUDA和cudnn上的步骤。

所以该[Mask RCNN](https://github.com/matterport/Mask_RCNN)的环境配置可总结如下：
* 首先在Ubuntu本地安装相应版本的CUDA和cudnn
* 将`requirements.txt`中的`tesorflow`改为`tensorflow-gpu>=1.3.0`
* 将该仓库下载到本地
* 新建conda环境，在环境中执行下面三条命令：
  * pip install -r requirements.txt
  * python3 setup.py install
  * pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

通过上述步骤即可完成环境配置。

该项目里还提到了`5K minival`和`35K validation-minus-minival`，这是由MS COCO的val分成的两个子集，使用`python3 samples/coco/coco.py evaluate --dataset=/path/to/coco/ --model=mask_rcnn_coco.h5`测试模型时，默认使用的是`5K minival`这5000张图片。

> */path/to/coco/* 指向MS COCO数据集的根目录

#### Reference
[Mask RCNN](https://github.com/matterport/Mask_RCNN)
