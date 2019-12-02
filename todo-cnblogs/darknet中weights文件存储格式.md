> 以下内容根据个人理解整理而成，如有错误，欢迎指出，不胜感激。
### 0. 写在前面
本文对darkent保存的`.weights`文件进行分析，以便后续将权值进行导出。
* 复习所涉及的c语言知识：sprinf(), fwrite()&fread(), FILE类型
* .weights中权值的存储格式

### 1. sprinf(), fwrite()&fread(), FILE类型
**sprinf():**
sprinf将一个格式化的字符串输出到一个目的字符串buff中：
```
// 在darknet中的使用
char buff[256];
sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
```

**fwrite()&fread():**
fwrite以二进制方式向文件流中写入数据：
```
// buffer: 数据源地址
// size:  每个单元字节数
// count: 总计单元数
// stream: 文件流指针
size_t fwrite(void* buffer, size_t size, size_t count, FILE * stream);

// 在darknet中的使用
fwrite(l.weights, sizeof(float), num, fp);
```

fread与fwrite类似，只不过buffer变为目的地址
```
size_t fread(void* buffer, size_t size, size_t count, FILE * stream);
```

**FILE:**
使用fopen( )函数可以创建一个新的文件或者打开一个已有的文件，这个调用会初始化一个FILE类型的对象，FILE类型包含了所有用来控制流的必要的信息。
```
FILE *fp = fopen(filename, "wb");
int b = fclose( FILE *fp );
```

### 2. .weights中权值的存储格式
> 以下内容主要以conv层为例

在`detector.c`的`train_detector()`函数末尾，保存权值的相关代码如下：
```
// buff只是一个字符串，并没有实际创建文件
char buff[256];
sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);

save_weights(net, buff);
```

通过向上追溯，可以在`parser.c`中找到函数`save_weights()`，并进一步追踪到`save_weights_upto()`，主要代码注释如下：
```
// *filename: 即为前面的buff字符串
// cutoff: 网络层数
void save_weights_upto(network net, char *filename, int cutoff)
{
    // 初始化一个文件读写流
    FILE *fp = fopen(filename, "wb");
    if(!fp) file_error(filename);

    // 以下三个变量在version.h中定义
    // #define MAJOR_VERSION 0
    // #define MINOR_VERSION 2
    // #define PATCH_VERSION 5
    int major = MAJOR_VERSION;
    int minor = MINOR_VERSION;
    int revision = PATCH_VERSION;

    fwrite(&major, sizeof(int), 1, fp);
    fwrite(&minor, sizeof(int), 1, fp);
    fwrite(&revision, sizeof(int), 1, fp);
    // net.seen用于记录训练时一共经历了多少张图片
    // 可根据该参数及cfg中对batch的配置，得出当前迭代次数
    fwrite(net.seen, sizeof(uint64_t), 1, fp);

    // 逐层保存权值
    int i;
    for(i = 0; i < net.n && i < cutoff; ++i){
        layer l = net.layers[i];
        if(l.type == CONVOLUTIONAL && l.share_layer == NULL){
            save_convolutional_weights(l, fp);
        } if(l.type == CONNECTED){
            save_connected_weights(l, fp);
        } if(l.type == BATCHNORM){
            save_batchnorm_weights(l, fp);
        }
    }
    fclose(fp);
}
```

在具体分析`save_convolutional_weights()`函数之前，首先要分析`convolutional_layer.c`中的`make_convolutional_layer()`函数，该函数根据每个卷积层的配置，为当前层参数分配相应数量的内存，主要代码注释如下：
```
// 卷积核个数
l.n = n;

// 卷积核权重总个数：n*c*size*size  groups是分组卷积时的参数，默认为1
l.nweights = (c / groups) * n * size * size;

// 为卷积核权值、偏置、BN参数分配内存
l.weights = (float*)calloc(l.nweights, sizeof(float));
l.biases = (float*)calloc(n, sizeof(float));
l.scales = (float*)calloc(n, sizeof(float));
l.rolling_mean = (float*)calloc(n, sizeof(float));
l.rolling_variance = (float*)calloc(n, sizeof(float));
```

再来看`parser.c`中的`save_convolutional_weights()`函数就比较容易理解：
```
void save_convolutional_weights(layer l, FILE *fp)
{
    int num = l.nweights;

    // 有BN的卷积层应该是不需要bias的
    fwrite(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.n, fp);
        fwrite(l.rolling_mean, sizeof(float), l.n, fp);
        fwrite(l.rolling_variance, sizeof(float), l.n, fp);
    }
    fwrite(l.weights, sizeof(float), num, fp);
}
```

### 3. 总结
从以上分析可以看出，`.weights`文件实际上就是一个字节流，我们只需要根据其保存时的顺序，每次读取相应字节数量的内容即可将其解析出来。

要注意，对卷积权重，这里是一维形式进行存储，就相当于将一个`N*C*H*W`的tensor展开成一维向量。

### Reference
[code](https://github.com/AlexeyAB/darknet)
