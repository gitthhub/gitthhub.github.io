---
layout:     post
title:      Deep Learning - Optimization
subtitle:   深度学习基础
date:       2019-04-01
author:     vhpg
header-img: img/placeholder_img.png
catalog: true
tags:
    - Machine Learning
    - Deep Learning
    - Optimization
    - Mathematics
---
> 本篇文章观点仅限于目前的理解，后续若有新的理解，还会继续更新。
> 文中编辑的公式，在网页上查看会有问题，可使用Atom+MarkDown插件查看(KaTex)

#### 0. Introduce
  本文对神经网络训练时常用的优化器进行总结，限于个人水平有限，无法从一个大的角度来谈深度学习中的优化问题，因此本文内容主要摘自[这篇文章](http://ruder.io/optimizing-gradient-descent/)，并加上一些自己的理解。

  总的来说，深度学习中的优化方法都是基于梯度下降的，都是对梯度下降方法的改进。所以本文首先介绍梯度下降算法及相应的变种，然后简单阐述在深度学习训练过程中的一些挑战，再介绍当前深度学习训练中一些流行的算法和其在Keras中的配置。


#### 1. Gradient descent and its variants




#### 1. SGD

##### 原理


[Reference](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

#### 2. RMSprop



[Reference](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)


#### 3. Adagrad


#### 4. Adam



#### 5. Adadelta


#### 6. Adamax


#### 7. Nadam


#### Reference
[Blog 1](http://ruder.io/optimizing-gradient-descent/)

[Keras](https://keras.io/optimizers/)

[CS231n]()
