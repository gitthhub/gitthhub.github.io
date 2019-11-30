---
layout:     post
title:      OpenCV Notes
subtitle:   OpenCV 使用笔记
date:       2019-08-21
author:     vhpg
header-img: img/placeholder_img.png
catalog: true
tags:
    - OpenCV
---

> 记录OpenCV在使用过程中遇到的一些问题(默认为cv2)。

#### 1. 图像缩放 resize
* 大图像变为小图像：将图像按小图像大小划分成相应数量的小格，每个格点内的像素求均值作为新图像的像素值；
* 小图像变为大图像：使用插值方法，包括最近邻插值、双线性插值（默认）、双三次插值、基于局部像素的重采样、Lanczos插值等。

* **最近邻插值**：$X_{src} = X_{dst} * (W_{src}/W_{dst})$

* **双线性插值**


#### 2. 仿射变换 warpAffine
