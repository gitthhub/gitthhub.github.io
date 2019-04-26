---
layout:     post
title:      Reproduce - AlexNet
subtitle:   论文复现
date:       2019-04-26
author:     vhpg
header-img: img/placeholder_img.png
catalog: true
tags:
    - Deep Learning
---
ImageNet中的图片是以按照一种语义网络(WordNet)的方式组织的，可以把WordNet理解为一种层次结构，每一种物体根据语义的大小可以被分为不同的类别(个人理解：人既可以被分为`人`，也可以被分为`哺乳动物`，而`哺乳动物`的语义相对大一些)，大的语义节点包含一些小的语义节点。

在ILSVRC-2012中，共使用了1000类物体，这1000类物体类别使用整数`1-1000`表示。严格地说，应该是1000类synsets(`sets of synonymous nouns`，近义词名词集合，如`钱包`在ILSVRC中被视为一类，但该类别相近的名词可以为`wallet, billfold, notecase, pocketbook`)，它相当于是ImageNet的一个子集，每一类都是ImageNet中的一个低级节点，且这些类别之间没有交集，并且没有语义包含关系。

ILSVRC竞赛中，每个synsets的信息都被存储在synsets数组中，这些数组被打包为meta.mat文件，每个synsets的信息存储格式如下:
```

```


array([(array([[1]], dtype=uint8),
        array(['n02119789'], dtype='<U9'),
        array(['kit fox, Vulpes macrotis'], dtype='<U24'),
        array([ 'small grey fox of southwestern United States; may be a subspecies of Vulpes velox'], dtype='<U81'), 
        array([[0]], dtype=uint8),
        array([], shape=(1, 0), dtype=uint8),
        array([[0]], dtype=uint8),
        array([[1300]], dtype=uint16))],

        dtype=[('ILSVRC2012_ID', 'O'), ('WNID', 'O'), ('words', 'O'), ('gloss', 'O'), ('num_children', 'O'), ('children', 'O'),      ('wordnet_height', 'O'), ('num_train_images', 'O')])
