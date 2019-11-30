---
layout:     post
title:      Image Quality Assessment Base
subtitle:   图像质量评估基础-数据集/评估标准
date:       2019-09-12
author:     vhpg
header-img: img/placeholder_img.png
catalog: true
tags:
    - Image Quality Assessment
---

#### Introduce
质量评估有图像质量评价(Image Quality Assessment, IQA)和视频质量评估(Video Quality Assessment, VQA)，QA右可分为主观方法(人直接判断)和客观方法(算法判断)，客观质量评估方法又分为全参考方法(Full-Reference, FR)，半参考方法(Reduced-Reference, RR)和无参考方法(No Reference-IQA, NR-IQA), 无参考也叫盲参考(Blind IQA, BIQA)，BIQA相对较难，也较为实用，是近些年研究的热点。

#### 数据集
如下图所示，黑体标出的是公认度较高的数据集(Subjects No表示参与评价的人数)：
![v2-52704ccf942da983ad2344322779a626_r](/assets/v2-52704ccf942da983ad2344322779a626_r.jpg)

每个数据集都给出了图像的平均主观得分(MOS)。

其中LIVE和CSIQ数据集主要针对常见失真类型，如加性高斯白噪声、高斯模糊、JPEG压缩和JPEG2000压缩等，TID2008和TID2013覆盖的失真类型则较为广泛，TID2013目前较难。

##### TID2013
TID2013包含25张参考图像，提供每张图像的24种失真和5个级别的失真水平，总计3000张失真图像，所有图像均以BMP格式保存，无任何压缩。
文件命名方式为：iXX_YY_Z.bmp
* XX: 参考图像号，[1-25]
* YY: 失真类型，[1-24]
* Z: 失真水平，[1-5]

每幅图像的平均主观得分(MOS)由来自五个国家的971个实验者给出，MOS(0-9)的值越大，图像质量越好。
TID2013公开了每幅图像的MOS值和相应的标准差。

##### TID2008
TID2013可以看做是TID2008的扩展，它们的参考图像是一样的，只不过TID2008只有17种失真类型，每种失真类型仅有4个级别的失真水平，总计1700张失真图像。

##### LIVE
区别LIVE数据集和LIVE In the Wild Image Quality Challenge Database(比赛用)
LIVE数据集包含5种失真类型，每种失真类型有7到8种失真程度

LIVE In the Wild Image Quality Challenge Database中的图片全部来自自然图片，没有经过人工引入失真，每张图片都有相应的MOS。

##### CSIQ
CSIQ数据集包含6种失真类型，每种失真类型有3到5种失真程度

#### 评估指标
在图像质量评估中，评价算法性能好坏的方法就是看在具有不同失真度的数据集上，观察者的主观评分和算法评分的相关度，相关度越高说明算法越好。

在图像质量评估中，常用的表示相关度的指标为`PLCC/SRCC`，此外还有`KROCC/RMSE`等。

##### Pearson线性相关系数：PLCC/LCC
`Pearson linear correlation coefficient，PLCC`
PLCC描述了主观评分和算法评分之间的线性相关性(绝对值越大越好)，当两个变量存在线性关系时，PLCC为1或-1，计算如下：
$$PLCC = \frac{\sum_{i=1}^{N}(y_i-\bar{y})(\hat{y_i}-\bar{\hat{y}})}{\sqrt{\sum_{i=1}^{N}(y_i-\bar{y})^2}\sqrt{\sum_{i=1}^{N}(\hat{y_i}-\bar{\hat{y}})^2}}$$
其中，$N$表示失真图像个数，$y_i$和$\hat{y_i}$分别表示第$i$幅图像的主观评分值和算法预测值，$\bar{y}$和$\bar{\hat{y}}$分别表示主观评分值得均值和预测值的均值。

##### Spearman秩相关系数：SRCC/SROCC
`Spearman rank-order correlation coefficient，SROCC`
SRCC用于度量变量之间关系的强弱(绝对值越大越好)。在没有重复数据的情况下，如果一个变量是另外一个变量的严格单调函数，则SRCC为1或-1，称完全秩相关，计算如下：
$$SRCC = 1-\frac{6\sum_{i=1}^{N}(v_i-p_i)^2}{N(N^2-1)}$$
其中，$v_i$和$p_i$分别表示$y_i$和$\hat{y_i}$在真实值和预测值序列中的排列位置。

##### Kendall秩相关系数：KRCC/KROCC
`Kendall rank-order correlation coefficient，KROCC`
KRCC和SRCC类似，衡量了算法预测的单调性。

##### 均方根误差：RMSE
`Root Mean Squared Error, RMSE`
RMSE和PLCC类似，评价算法预测的准确性，但RMSE越小越好，最小为0。


#### Reference
[图像质量评估综述](https://zhuanlan.zhihu.com/p/32553977)
[全参考图像质量评价方法整理与实用性探讨](https://zhuanlan.zhihu.com/p/24804170)
