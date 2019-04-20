---
layout:     post
title:      CNN Model - MobileNet v1&v2
subtitle:   论文分析
date:       2019-04-20
author:     vhpg
header-img: img/placeholder_img.png
catalog: true
tags:
    - Deep Learning
    - CNN Model
---
> 本篇文章观点仅限于目前的理解，后续若有新的理解，还会继续更新。

#### 1. Introduce
本文涉及两篇论文，分别是2017年的MobileNet v1(未发文章)和2018年CVPR的MobileNet v2。
MobileNet提出的目的是通过优化网络结构，使得在保证一定精度的情况下尽可能减少网络参数和降低计算量，以便在嵌入式设备中运行。

MobileNet v1比较简单，仅仅是深度可分离卷积，MobileNet v2则稍微复杂一点。
本文简述MobileNet v1的思路，并重点介绍MobileNet v2。


#### 2. MobileNet v1
如果读了Xception，觉得MobileNet v1并无新颖之处。MobileNet v1中基本所有的卷积使用的都是`depth-wise convolution`深度可分离卷积，通过作者在论文中的分析可发现，这种`depth-wise convolution`可大大降低参数数量和计算量，这里不再赘述。

另外，作者引入了两个超参数用于控制网络规模，这两个超参数分别用于控制feature map的channel和width相对基础模型的比例，进而控制网络整体的规模，这种思路在前面有些网络的设计中也有使用(暂时忘记是哪个网络了)。

MobileNet v1使用TensorFlow实现起来也并不麻烦，本文给自己的启发应该是实验的设计，这样一个简单的想法，如何通过设计出有对比性的实验来评价模型的性能并得出一系列有说服力的结论。

#### 3. MobileNet v2





#### Reference
[MobileNet v1](https://arxiv.org/pdf/1704.04861.pdf)
[MobileNet v2](https://arxiv.org/pdf/1801.04381.pdf)
