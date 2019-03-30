---
layout:     post
title:      CNN Model - Inception v3: Rethinking the Inception
subtitle:   论文分析
date:       2019-03-29
author:     vhpg
header-img: img/placeholder_img.png
catalog: true
tags:
    - Deep Learning
    - CNN Model
---
> 本篇文章观点仅限于目前的理解，后续若有新的理解，还会继续更新。

#### 1. Introduce
  Inception v3是由Szegedy于2016年发表在CVPR上的文章[Rethinking the Inception Architecture for Computer Vision](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)中提出的，这也是该作者对Inception结构的再一次优化。

  作者的motivation很简单，较大的网络能够带来更好的性能，却造成计算量的增加，所以能否通过优化网络结构来达到增大网络但参数数量和计算量不太大的目的。

  论文中的实验评估是基于ILSVRC-2012 classification任务的validation set：
  ```
                top-1     top-5
    1 model     21.2%     5.6%
    4 models    17.3%     3.5%
  ```

  到目前为止，一些典型网络的对比：
  ```
    Name          Parameters     layers      Multiply-Adds
  * AlexNet       60  million   8  layers
  * Inception v1  5   million   22 layers     1.5 billion
  * VGGNet        144 million   19 layers
  * Inception v2  25  million   42 layers     4   billion
  ```
  引用[这里](https://blog.csdn.net/u010402786/article/details/52433324)的一张图片做个简单的小结，这张图总结的并不完整，等这方面的论文看得差不多了再进行更新：
  ![2019-03-30_092151](/assets/2019-03-30_092151.png)

#### 2. General Design Principles
  作者指出，在Inception v1论文中，并没有给出一种有效的使用Inception v1构建其他网络的方法，这给将该结构用于其他应用带来一定的困难，所以这里作者给出了一些一般的设计原则，这些原则并非可以直接使用，但是可以在提高网络性能遇到问题时考虑使用：
  * 避免表达瓶颈，尤其是在较浅层的网络。在CNN网络中，信息从输入流向输出，我们从任一层切开，都可以获取通过该层的信息，我们要尽量避免在某一层对信息的过度压缩，特征表达的大小应该慢慢地下降。信息容量不能仅使用表达的维度来衡量。`理解为：相邻层feature map的size不要一下子降得太多，这里的size指w,h,c。`

  * 高维表达更易在网络的局部进行处理，增加每个卷积核的激活单元可以获取更加松散的特征，这会加速网络训练。`这段话不太好翻译，理解为：在高层应增大卷积核的大小。`

  * 在低维嵌入中可以做空间聚集(指的应该是压缩
  * )而不会有太多表达能力上的损失，这应该是得益于相邻单元的比较强的相关性，并且可加快训练。`理解为：在使用正常的卷积核进行卷积前，可使用1*1的卷积核对feature map进行维度的压缩而不影响特征表达能力。`

  * 平衡网络的宽度和深度。平衡每一步的卷积核的数量和网络深度可达到网络的最优性能。同时增加网络的宽度和深度有助于提高网络性能，但是对一个计算量确定的任务，同时增加这两项可能达到性能饱和。所以计算负担应该使用一种平衡的方式分布于网络的宽度和深度上。`理解为：对网络的各层，应该通过调整宽度和深度使得计算量均匀分布，如增大深度的同时减小宽度，YOLOv3网络的设计中，作者就是尽量使每一层的计算量相同。`

#### 3. Factorizing Convolutions with Large Filter Size
  本文中，作者给出了一种分解较大卷积核的方法，可降低参数数量。
  如下图所示，对一个`5*5`的卷积核的卷积结果，可看做是由连续两层的`3*3`的卷积核卷积得到的：
  ![2019-03-30_101227](/assets/2019-03-30_101227.png)

  下图是Inception v1中的Inception模块：
  ![2019-03-30_101508](/assets/2019-03-30_101508.png)
  其中`5*5`的卷积核被代替后如下图所示，记为**Figure 5**:
  ![2019-03-30_101520](/assets/2019-03-30_101520.png)

  如下图所示，进一步考虑对`3*3`卷积核的分解，可发现它可看做由连续两层的`1*3`和`3*1`的卷积核卷积得到：
  ![2019-03-30_101811](/assets/2019-03-30_101811.png)

  原则上，任意一个`n*n`的卷积核都可由两个`1*n`和`n*1`的卷积核代替，并且当n越大，节约的计算量就越大。所以原始的Inception结构可更改为如下结构，记为**Figure 6**：
  ![2019-03-30_102341](/assets/2019-03-30_102341.png)

  作者通过实验发现，在较浅的层应用该分解，效果并不好，但是在中等大小(12-20)的feature map上效果很好。在作者给出的网络结构中，对`17*17`的feature map使用`n=7`。

  对于较粗糙的feature map`(8*8)`，作者使用了如下结构以提升高维表达能力(**根据principle 2？**)，记为**Figure 7**:
  ![2019-03-30_103146](/assets/2019-03-30_103146.png)

#### 4. Utility of Auxiliary Classifiers
  作者对Inception v1中提出的辅助分类器进行了进一步的分析：
  使用辅助分类器最初的动机是为了应对梯度消失问题，想把梯度信息较快传到较低层并加速收敛。
  进一步的实验发现：
  * 在训练早期，网络没有达到较高精度时，有没有辅助分类器的网络的收敛曲线几乎一致，所以辅助分类器在训练早期并不能加速收敛；

  * 在训练接近尾声，网络精度较高时，有辅助分类器的网络比没有辅助分类器得网络的最终精度要高；

  * 当时使用了两个辅助分类器，实验发现，移除较低的辅助分类器对最终结果没有影响，所以它也就没有之前所说的发展底层特征的作用；

  * 现在我们发现，辅助分类器的行为实际上相当于正则化，因为当分支是BN或Dropout层时，主分类器的性能更好。

#### 5. Efficient Grid Size Reduction and Network Structure
  这一部分，作者主要探讨feature map的size的问题。
  CNN中一般使用Pooling操作来降低feature map的size，但是作者认为直接对feature map进行Pooling操作会遇到前面所述的表达瓶颈的问题，`d * d * k -> d/2 * d/2 * k`。
  作者认为应该在Pooling操作前使用`1*1`的卷积核对feature map进行升维，即:`d * d * k -> d * d * 2k -> d/2 * d/2 * 2k`，但这样由于卷积升维操作带来的计算量会很大`2*d*d*k*k`。
  另一种操作是先进行Pooling然后再使用`1*1`的卷积核进行升维，即:`d * d * k -> d/2 * d/2 * k -> d/2 * d/2 * 2k`，相比于前一种，这样的计算量降低1/4，但是作者认为池化后的参数数量变为了`d/2 * d/2 * k`，这里会有表达瓶颈。

  以上两种方法的简单示意图如下：
  ![2019-03-30_112554](/assets/2019-03-30_112554.png)

  作者提出了一种新的方法来解决上述问题，即同时进行步长为2的卷积核池化操作，再将其feature map进行联合，如下图所示，这也是作者在网络中采用的Pooling方案：
  ![2019-03-30_112801](/assets/2019-03-30_112801.png)

  作者提出的网络的整体结构如下表所示，表中没有标明`padded`的卷积层没有使用`0-padding`，由表中还可以看出，作者使用了多种方式来降低feature map的size：
  * 步长为2的卷积；
  * 上述的卷积池化并行的池化操作；
  * 最后一层使用的是`8*8`的池化操作；
  ![2019-03-30_103722](/assets/2019-03-30_103722.png)
  另外需要说明：
  * 该网络结构被作者称为`Inception v2`；
  * 后面有根据该网络结构的一些改进，最优改进`Inception v2 + BN-auxiliary`被作者称为`Inception v3`；

#### 6. Model Regularization via Label Smoothing
  作者提出了一个正则化分类层的机制：在训练阶段通过估计标签失活的边缘影响来正则化分类器层。
  这部分没看懂，后续再补充。

#### Reference
[Inception v3: Rethinking the Inception](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)

[图片来源](https://blog.csdn.net/u010402786/article/details/52433324)
