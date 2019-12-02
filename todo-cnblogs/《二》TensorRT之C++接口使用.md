> 以下内容根据个人理解整理而成，如有错误，欢迎指出，不胜感激。
### 0. 写在前面
本文首先根据[TensorRT开发者指南](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#initialize_library)梳理TensorRT的C++接口使用流程，然后基于TensorRT的官方例程[“Hello World” For TensorRT](https://github.com/NVIDIA/TensorRT/blob/release/6.0/samples/opensource/sampleMNIST/README.md)来了解其具体使用方式。

### 1. C++接口使用
由[上一篇Blog](https://www.cnblogs.com/vh-pg/p/11677137.html)中的内容可知，模型从导入TensorRT到执行Inference大致经过下面三个阶段：
* Network Definition
* Builder
* Engine
这三个阶段分别对应着TensorRT中一些重要的类和方法，下面分别来叙述。

> `ILogger`

首先说明一个必须但不是很重要的类`ILogger`，它用于记录一些日志信息。
在编程时，我们需要声明一个全局的`ILogger`对象gLogger，TensorRT中很多方法都需要它作为参数
(貌似需要继承`ILogger`类来编写自己的Logger类)

> `IBuilder`

`IBuilder`类应该算是最重要的一个类，在使用时，首先要使用TensorRT的全局方法`createInferBuilder()`来创建一个`IBuilder`类指针，然后由该指针调用`IBuilder`类方法创建Network和Engine类的指针。

> `INetworkDefinition`

`INetworkDefinition`类即为网络定义，可通过`IBuilder`类方法`createNetwork()`返回其指针。

> `ICudaEngine`

`ICudaEngine`类即为Engine，可通过`IBuilder`类方法`buildCudaEngine()/buildEngineWithConfig()`返回其指针。
注意，可通过导入模型生成Engine和通过反序列化来加载Engine两种Engine生成方式。

> `IParser`

`IParser`类对应着前文所述的三种不同的解释器，可根据需要来使用。
`IParser`类方法`parse()`用于解析并加载模型及参数到TensorRT网络中(`INetworkDefinition`类)

> `IExecutionContext`

Engine的运行需要一个运行时环境，`createExecutionContext()`方法为相应的`ICudaEngine`生成一个`IExecutionContext`类型的运行环境context。

一个简单的代码示例如下：
```
# builder
IBuilder* builder = createInferBuilder(gLogger);

# network
INetworkDefinition* network = builder->createNetwork();

# parser -> load params to network
CaffeParser* parser = createCaffeParser();
const IBlobNameToTensor* blobNameToTensor = parser->parse(args);

# engine
builder->setMaxBatchSize(maxBatchSize);
IBuilderConfig * config = builder->createBuilderConfig();
config->setMaxWorkspaceSize(1 << 20);
ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

# serialize engine if necessary
IHostMemory *serializedModel = engine->serialize();

//# 如果是直接反序列化来获取engine，上述很多步骤都不需要
//IRuntime* runtime = createInferRuntime(gLogger);
//ICudaEngine* engine = runtime->deserializeCudaEngine(modelData, modelSize, nullptr);

# inference
IExecutionContext *context = engine->createExecutionContext();

# 获取输入输出层的索引
int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);

# 指针指向输入输出层在GPU中的存储位置
void* buffers[2];
buffers[inputIndex] = inputbuffer;
buffers[outputIndex] = outputBuffer;

# 异步执行inference
context->enqueue(batchSize, buffers, stream, nullptr);

# clear
serializedModel->destroy();
parser->destroy();
network->destroy();
config->destroy();
builder->destroy();
...
```

**疑问一：IBuilder配置参数**
可根据需要来配置builder，其中比较重要的参数有两个：
* maxBatchSize: TensorRT的输入是NHWC格式，maxBatchSize表明了N最大可为多少，如当N=8时，模型一次可处理8张图片，速度要大于调用8次、每次处理一张图片的总时间；
* maxWorkspaceSize：每一层算法的运行都需要临时的存储空间，该参数限制了每一层能够使用的最大临时存储空间。

**疑问二：runtime vs context**
在读开发者指南时，这两个概念有点乱，这里区分一下：
* runtime: 直接反序列化获取engine时，需要定义该类，此时不需要前面builder等相关的操作
* context：每一个engine的运行都需要context，一个engine可有多个context

**注意一：CUDA context**
从开发者指南中可知，尽管可以使用默认的CUDA context而不显式创建它，但官方不建议这么做，推荐在创建一个runtime或builder对象时，创建并配置自己的CUDA context。

### 2. [“Hello World” For TensorRT](https://github.com/NVIDIA/TensorRT/blob/release/6.0/samples/opensource/sampleMNIST/README.md)
理清上述流程后，该例子的cpp源码不难理解，具体细节这里不再阐述。
