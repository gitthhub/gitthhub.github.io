---
layout:     post
title:      TensorFlow & Keras
subtitle:   使用笔记
date:       2019-04-04
author:     vhpg
header-img: img/placeholder_img.png
catalog: true
tags:
    - Deep Learning
    - TensorFlow
---
> 本篇文章是在使用TensorFlow和Keras过程中的一些笔记，持续更新。

### 1. 常用方法
##### tf.meshgrid()
如下面的例子，相当于将x作为行向量按照y的大小复制相应的行数，将y作为列向量按照x的大小复制相应的列数，得到的X和Y的shape为(size(y), size(x)):
```
  x = tf.range(5)
  y = tf.range(3)
  X, Y = tf.meshgrid(x,y)

  output:
  X: shape = (3, 5)
    [[0, 1, 2, 3, 4],
     [0, 1, 2, 3, 4],
     [0, 1, 2, 3, 4]]

  Y: shape = (3, 5)
    [[0, 0, 0, 0, 0],
     [1, 1, 1, 1, 1],
     [2, 2, 2, 2, 2]]
```

##### tf.reshape(a, shape)
跟据Jupyter notebook中输出的结果，tf中的tensor实际存储为Numpy中的array，参考《利用Python进行数据分析》第12章-高级数组操作中的内容，可总结如下：
* 传入的shape参数有一维可以为-1，则该维的大小由数据本身的shape推断得到；
* Numpy数组是按行优先的顺序进行创建和存储，与之相对的有列优先顺序，二者的最主要区别是，在对数组reshape的过程中，行优先顺序是先处理较高的维度，如轴1先于轴0被处理。从对Numpy的轴的理解来看，更高维(最后一维)实际上对应的就是某一行元素的索引。
