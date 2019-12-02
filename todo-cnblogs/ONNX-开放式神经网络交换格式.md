> 以下内容根据个人理解整理而成，如有错误，欢迎指出，不胜感激。

### 1. ONNX简介
>ONNX是一种针对机器学习所设计的开放式的文件格式，用于存储训练好的模型。它使得不同的人工智能框架（如Pytorch, MXNet）可以采用相同格式存储模型数据并交互。 ONNX的规范及代码主要由微软，亚马逊 ，Facebook 和 IBM 等公司共同开发，以开放源代码的方式托管在Github上。目前官方支持加载ONNX模型并进行推理的深度学习框架有： Caffe2, PyTorch, MXNet，ML.NET，TensorRT 和 Microsoft CNTK，并且 TensorFlow 也非官方的支持ONNX。---维基百科

ONNX全称是Open Neural Network Exchange，不同深度学习框架可以将模型保存为ONNX格式，从而实现模型在不同框架之间的转换。
ONNX中，每一个计算流图都定义为由节点组成的列表，每个节点是一个OP，可能有一个或多个输入与输出，并由这些节点构建有向无环图。
目前，ONNX已支持当前主要的各种深度学习框架，有些框架如PyTorch是官方集成了ONNX，有些需要第三方支持，即便像darknet这种小众的框架，也可以手动构建ONNX图来将模型转为ONNX格式。

我们可以使用pip或在conda环境中使用conda来获取ONNX，具体参见[ONNX的github](https://github.com/onnx/onnx)。

### 2. ONNX使用
ONNX是一个开放式规范，由以下组件组成：
* 可扩展计算图模型
* 一系列内置运算单元(OP)
* 标准数据类型

将一个模型转为ONNX格式，主要是构造计算流图，在官方github的[example](https://github.com/onnx/onnx/tree/master/onnx/examples)目录下有很多使用示例，这里列举出最常用的几个方法。

在ONNX中，数据的存储使用的是Google的Protobuf序列化框架，数据结构主要有以下六种，定义在`onnx/onnx.in.proto`文件内：
```
TensorProto
ValueInfoProto
AttributeProto

NodeProto
ModelProto
GraphProto
```

构造上述数据类型的方法定义在`onnx/helper.py`文件内，具体如下：
```
helper.make_tensor()            
helper.make_tensor_value_info()  
helper.make_attribute()         

helper.make_node()  
helper.make_model()   
helper.make_graph()
```

另外两个比较常用的方法是对构造完的模型的检查和保存，分别定义在`onnx/checker.py`和`onnx/__init__.py`中：
```
onnx.checker.check_model()
onnx.save()
```

下面给出上述主要方法的定义，具体使用示例可参考官方[example](https://github.com/onnx/onnx/tree/master/onnx/examples)：
**helper.make_tensor()**
```
def make_tensor(
        name,       # tensor名称(string)
        data_type,  # tensor内数据类型(TensorProto.dataType)
        dims,       # tensor的shape(list of int)
        vals,       # tensor的值
        raw=False   # 当为false时，该方法会根据data_type类型来存储vals，当为true时，该方法会使用raw_data来存储vals(在该方法中指的是bytes类型) -> 此处理解存疑(proto field ?)
):  # type: (...) -> TensorProto
```

**helper.make_tensor_value_info()**
```
# 通常和make_tensor()一起使用，用于创建一个tensor的信息
def make_tensor_value_info(
        name,                   # tensor名称(string)
        elem_type,              # tensor中元素的类型(TensorProto.dataType)
        shape,                  # tensor的shape(list of int)
        doc_string="",          # 可选参数：对该tensor的描述
        shape_denotation=None,  # 可选参数：对shape中每个维度的描述(list of string)
):  # type: (...) -> ValueInfoProto
```

**helper.make_attribute()**
```
# 该方法是make_node()的内部调用方法，当我们通过**kwargs传入一系列op的属性时，最终都会调用该方法转化为键值对，因此可以直接调用该方法先构造键值对，然后将构造结果传给**kwargss
def make_attribute(
        key,             
        value,           
        doc_string=None  
):  # type: (...) -> AttributeProto
```

**helper.make_node()**
```
def make_node(
        op_type,          # 要构造的op的名字(string) -> 相当于是将一个op封装为一个节点
        inputs,           # 输入当前节点的节点名称(list of string)  
        outputs,          # 当前节点输出的名称(list of string)  -> 当节点只有一个输出时，节点名称就相当于outputs
        name=None,        # 可选参数：当前节点的名称，作为索引该节点的唯一标识(string)
        doc_string=None,  # 可选参数：为当前节点添加描述(string)
        domain=None,      # 可选参数：为当前节点添加一个领域？(string)
        **kwargs          # 当前节点的属性(dict or 类似普通参数的传入)，不同类型的op具有不同的属性，具体参数需要看构造的op类型，然后参考官方op文档所给的属性选项
):  # type: (...) -> NodeProto
```

**helper.make_graph()**
```
def make_graph(
    nodes,             # 节点的列表(list of node)
    name,              # graph的名称(string)
    inputs,            # 输入网络的tensor的相关信息(list of ValueInfoProto)，包括网络每层的权重等信息 -> 由make_tensor_value_info()构造
    outputs,           # 网络输出tensor的相关信息(list of ValueInfoProto) -> 由make_tensor_value_info()构造
    initializer=None,  # 可选参数：图中每个节点的初始化权值(list of TensorProto) -> 由make_tensor()构造
    doc_string=None,   # 可选参数：对graph的描述
    value_info=[],     # 可选参数：存放中间层产生的输出数据的信息(list of ValueInfoProto) -> 由make_tensor_value_info()构造
):  # type: (...) -> GraphProto
```

**helper.make_model()**
```
def make_model(
        graph, 
        **kwargs
):  # type: (GraphProto, **Any) -> ModelProto
```

### 3. 总结
总结一下构造一个onnx模型的具体流程：
1. 根据自己的网络结构调用`make_node()`来创建相关节点，节点的inputs和outputs参数决定了后续graph的连接情况，节点的权值和信息通过调用`make_tensor()`和`make_tensor_value_info()`来创建，它们和节点的联系在于节点的`name`
2. 上述三个方法构造的结构分别对应`make_graph()`中的三个参数，具体如下所示：
    * nodes: make_node()
    * inputs: make_tensor_value_info()
    * initializer: make_tensor()
3. 最后检查和保存模型即可

### Reference
[开源一年多的模型交换格式ONNX，已经一统框架江湖了？](https://www.jiqizhixin.com/articles/2018-11-30-6)
[ONNX github](https://github.com/onnx/onnx)
[ONNX example](https://github.com/onnx/onnx/tree/master/onnx/examples)
[ONNX op](https://github.com/onnx/onnx/blob/master/docs/Operators.md)