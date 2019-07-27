---
layout:     post
title:      Deep Learning - Darknet to Keras
subtitle:   darknet模型转换为Keras的.h5模型
date:       2019-06-03
author:     vhpg
header-img: img/placeholder_img.png
catalog: true
tags:
    - Deep Learning
---
> 本篇文章观点仅限于目前的理解，后续若有新的理解，还会继续更新。

#### 1. Darknet (.cfg & .weights)
darknet中，网络配置存储在`.cfg`文件中，权重等参数存储在`.weights`文件中，在`.weights`文件中，所有参数以二进制形式存储。下面根据darknet源码来分析参数的存储顺序。

网络训练的入口为`./examples/detector.c/train_detector()`，该函数调用了`save_weights()`函数来保存权重，`save_weights()`函数位于`./src/parser.c`中，通过逐层查找可发现，最终用于保存权重的是`save_convolutional_weights()`和`save_connected_weights()`等函数。

在`save_weights()`中，先保存了四项头文件，代码如下，前三个参数暂时不清楚表示什么含义，最后一个参数`net->seen`相当于`epoch`：

```
int major = 0;
int minor = 2;
int revision = 0;
fwrite(&major, sizeof(int), 1, fp);
fwrite(&minor, sizeof(int), 1, fp);
fwrite(&revision, sizeof(int), 1, fp);
fwrite(net->seen, sizeof(size_t), 1, fp);
```

`save_convolutional_weights()`的部分代码如下，其中`l.newights`为当前卷积层所有卷积权重的个数，`l.n`为当前卷积层卷积核的个数，`fwrite()`函数第一个参数指向`src`，最后一个参数指向`dst`。
可以看出，在卷积层，参数保存的顺序为`bias BN filters`，对于`filters`，它的`shape`为`[out_dim, in_dim, height, width]`，`out_dim`即为`l.n`，`in_dim`为输入的`channel`个数。

```
void save_convolutional_weights(layer l, FILE *fp)
{
    int num = l.nweights;
    fwrite(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.n, fp);
        fwrite(l.rolling_mean, sizeof(float), l.n, fp);
        fwrite(l.rolling_variance, sizeof(float), l.n, fp);
    }
    fwrite(l.weights, sizeof(float), num, fp);
}
```

全连接层的权重存储与此类似，存储顺序为`bias weights BN`。

根据以上分析，结合`.cfg`中网络的具体结构，通过依次读取相应的字节数，可分别取出各层的参数。

#### 2. Keras (.h5)
Keras的模型使用`hdf5`格式进行存储，`HDF`是一种为存储为存储和处理大容量科学数据而设计的文件格式和相应的库文件，当前较为流行的版本是`HDF5`。python中可使用`h5py`模块来操作`HDF5`数据。

Keras中提供了`get_weights()`函数查看数据：
```
for layer in model.layers:
  weights = layer.get_weights()  # list of numpy array
```
也可以通过`h5py`模块读取数据，具体操作代码如下：
```
import h5py

def print_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items())==0:
            return

        for layer, g in f.items():
            print("  {}".format(layer))
            print("    Attributes:")         # 权重名称及一些参数配置属性
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for p_name in g.keys():
                param = g[p_name]
                for k_name in param.keys():
                    print("      {}/{}: {}".format(p_name, k_name, param.get(k_name)[:]))
    finally:
        f.close()
```
输入大致如下：
```
layer_0
  Attributes:
    nb_params: 2
    subsample: [1 1]
    init: glorot_uniform
    nb_filter: 32
    name: Convolution2D
    activation: linear
    border_mode: full
    nb_col: 3
    stack_size: 3
    nb_row: 3
  Dataset:
    param_0: (32, 3, 3, 3)
    param_1: (32,)
layer_1
  Attributes:
    nb_params: 0
    activation: relu
    name: Activation
  Dataset:
layer_2
  Attributes:
    nb_params: 2
    subsample: [1 1]
    init: glorot_uniform
    nb_filter: 32
    name: Convolution2D
    activation: linear
    border_mode: valid
    nb_col: 3
    stack_size: 32
    nb_row: 3
  Dataset:
    param_0: (32, 32, 3, 3)
    param_1: (32,)
layer_3
  Attributes:
    nb_params: 0
    activation: relu
    name: Activation
  Dataset:
layer_4
  Attributes:
    nb_params: 0
    name: MaxPooling2D
    ignore_border: True
    poolsize: [2 2]
  Dataset:
```
可以看出，对keras来说，`hdf5`文件存储了包括网络配置和参数的所有信息：第一部分为文件的整体信息，第二部分为各层的配置`Attributes`和参数`Dataset`。

#### 3. Darknet to Keras
根据以上的分析，将darknet的模型转换为keras模型就比较简单，思路为，首先根据darknet的`.cfg`文件来逐层读取`.weights`权重，然后根据这些信息新建keras的层，使用`set_weights()`或直接在建层时将权重赋上去，最后将建立好的`model`保存为`.h5`模型。

具体的转换代码可参考[这里](https://github.com/qqwweee/keras-yolo3/blob/master/convert.py)，基本思路与上面的描述类似。
此处留有一个疑问，即在给`BN`层赋权重时，将卷积层的偏置参数也加进来了，此处应该是有问题的：
```
bn_weight_list = [
            bn_weights[0],  # scale gamma
            conv_bias,  # shift beta
            bn_weights[1],  # running mean
            bn_weights[2]  # running var
        ]
```

#### Reference
[.h5](https://github.com/keras-team/keras/issues/91)
[code](https://github.com/qqwweee/keras-yolo3/blob/master/convert.py)
