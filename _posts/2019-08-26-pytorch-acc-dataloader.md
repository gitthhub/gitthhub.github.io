---
layout:     post
title:      PyTorch - Accelerate DataLoader
subtitle:   PyTorch使用笔记
date:       2019-08-26
author:     vhpg
header-img: img/placeholder_img.png
catalog: true
tags:
    - PyTorch
---

#### Introduce
近期在使用PyTorch的过程中发现, PyTorch在图片加载和预处理上耗时较多,导致GPU的使用率波动较大,于是在网上搜集了一些加速DataLoader的方法.

#### Use NVIDIA DALI
来源: [How to speed up the data loader](https://discuss.pytorch.org/t/how-to-speed-up-the-data-loader/13740/13)

安装: `pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali`
github: `https://github.com/NVIDIA/DALI`
tutorial: `https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/examples/index.html`
doc: `https://docs.nvidia.com/deeplearning/sdk/dali-archived/dali_012_beta/dali-developer-guide/index.html`

NVIDIA DALI 是NVIDIA提供的用于数据预处理的pipline, 支持PyTorch/TensorFlow/MXNet等框架, 在数据加载的过程中可使用GPU进行加速.
PyTorch的transforms的所有操作是在CPU上完成, 然后再将Tensors复制到GPU上进行训练, 该操作耗时较多.

实测效果与官方的transform比,速度提升不明显(约10%):
  FER-2013, batch=256, epoch=1, GPU=0, 单GPU训练: 38s:43s
  FER-2013, batch=256, epoch=1, GPU=0, 双GPU训练: 27s:34s 调整dali的num_threads对速度基本无影响
  FER-2013, batch=64,  epoch=1, GPU=0, 双GPU训练: 41s:45s
在开始训练前, 官方的方法要等比较久的时间,而DALI开始的很快


#### Build an HDF5 file with all images
来源: [How to speed up the data loader](https://discuss.pytorch.org/t/how-to-speed-up-the-data-loader/13740/13)


#### torchvision.transforms.ToPILImage
来源: [How to speed up the data loader](https://discuss.pytorch.org/t/how-to-speed-up-the-data-loader/13740/13)


#### data prefetcher
来源: [知乎专栏](https://zhuanlan.zhihu.com/p/66145913)


#### 内存->硬盘 / 数据全部加载到内存
来源: [How to speed up the data loader](https://discuss.pytorch.org/t/how-to-speed-up-the-data-loader/13740/13)
