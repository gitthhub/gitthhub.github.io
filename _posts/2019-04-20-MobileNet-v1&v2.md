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
本文涉及两篇论文，分别是2017年的MobileNet v1(貌似是2015年就做出来，但是一直没放出来，后来才放到arXiv上)和2018年CVPR的MobileNet v2。
MobileNet提出的目的是通过优化网络结构，使得在保证一定精度的情况下尽可能减少网络参数和降低计算量，以便在嵌入式设备中运行。

MobileNet v1比较简单，仅仅是深度可分离卷积，MobileNet v2则稍微复杂一点。
本文简述MobileNet v1的思路，并重点介绍MobileNet v2。


#### 2. MobileNet v1
如果读了Xception，觉得MobileNet v1并无新颖之处。
1. MobileNet v1中基本所有的卷积使用的都是`depth-wise convolution`深度可分离卷积(类似Xception中，先`3*3`，再`1*1`)，通过作者在论文中的分析可发现，这种`depth-wise convolution`可大大降低参数数量和计算量，这里不再赘述。

2. MobileNet v1中的ReLU使用的是ReLU6 = min(max(feature, 0), 6)，使得输出限制在[0, 6]，在进行量化时，可以避免因为输出较大而造成的精度损失。

另外，作者引入了两个超参数用于控制网络规模，这两个超参数分别用于控制feature map的channel和width相对基础模型的比例，进而控制网络整体的规模，这种思路在前面有些网络的设计中也有使用(暂时忘记是哪个网络了)。

MobileNet v1使用TensorFlow实现起来也并不麻烦，本文给自己的启发应该是实验的设计，这样一个简单的想法，如何通过设计出有对比性的实验来评价模型的性能并得出一系列有说服力的结论。

#### 3. MobileNet v2
相比于MobileNet v1，MobileNet v2的主要改进是添加了线性Bottleneck和使用残差结构。

##### 3.1 Linear Bottlenecks
在Inception结构提出时，它的作者认为具有较多channel的feature map所包含的信息有冗余，可以使用`1*1`卷积将它们映射到较少channel的低维空间上，该`1*1`卷积层也通常被称为`Bottleneck`。

在这篇文章中，作者认为，如果在channel较少的低维张量中使用`1*1`卷积和ReLU，ReLU的使用会带来较大的信息损耗，例子如下：
![2019-04-21_085641](/assets/2019-04-21_085641.png)
在上图中，二维空间(也可能是三维空间)中的螺旋线所对应的张量使用一个`N*2`维的随机矩阵T将其映射到N维空间，在进行ReLU后使用T的逆将其映射回二维空间，可以发现，当N越小时，所造成的信息损失越严重。上述过程就相当于是将一个channel较少的低维张量映射到高维后再映射回低维，在高维中ReLU的使用造成了信息的损耗。

因此，在本文中作者提出，使用线性变换代替原来的`1*1`卷积中的激活层(即`1*1`卷积中不再使用激活函数)，而在需要使用激活层的张量中，首先使用较大的N将该张量扩张到高维，具体示意图如下：
![2019-04-21_091317](/assets/2019-04-21_091317.png)
(a)是常规的卷积操作；
(b)是常规的可分离卷积操作；
(c)是具有linear bottleneck的可分离卷积操作，与(b)相比的区别是，这里从低维到高维进行扩张时没有使用激活函数;
(d)是首先对低维张量进行没有激活函数的扩张，在高维空间中进行可分离卷积，最后再映射到所需大小。若是分别对(c)和(d)两种结构进行堆叠，可发现二者是等价的。

(以上内容还是稍有疑惑。)

##### 3.2 Inverted Residual
目前的网络结构设计中，残差结构已经成为必不可少的成分，在MobileNet v2中，作者也使用了该结构。
如下图所示，与普通的残差连接不同的是，作者这里将残差结构连接在两个低维张量间：
![2019-04-21_092242](/assets/2019-04-21_092242.png)

综合以上两点，作者给出MobileNet的基础网络结构Bottleneck residual block:
![2019-04-21_092756](/assets/2019-04-21_092756.png)

##### 3.3 Overall Architecture
MobileNet v2总体的网络结构如下，其中，t为扩张系数，c为channel数，n为重复次数，s为步长：
![2019-04-21_092944](/assets/2019-04-21_092944.png)

#### 4. Conclusion
深度可分离卷积使得网络性能在下降不大的情况下参数数量大量减少且速度有较大提升，MobileNet充分体现了这一优势，这里的实验部分不再赘述，后期会尝试复现其结果。



#### Reference
[MobileNet v1](https://arxiv.org/pdf/1704.04861.pdf)

[MobileNet v2](https://arxiv.org/pdf/1801.04381.pdf)

[Blog 1](https://blog.ddlee.cn/posts/c9816b0a/)
