---
layout:     post
title:      Numpy等科学计算库
subtitle:   使用笔记
date:       2019-04-04
author:     vhpg
header-img: img/placeholder_img.png
catalog: true
tags:
    - Numpy
---
> 本篇文章是在使用Numpy等Python科学计算库过程中的一些笔记，持续更新。

### 基础知识
##### array & asarray
array和asarray都可以将结构数据转化为ndarray，区别在于：
* 当数据源不是ndarray时，二者都对数据进行复制；
* 当数据源是ndarray时，asarray不再复制，而array仍会复制并返回副本；
