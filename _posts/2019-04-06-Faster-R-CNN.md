参考[1](https://zhuanlan.zhihu.com/p/31426458) [2](https://blog.csdn.net/shenxiaolu1984/article/details/51152614 [3](https://my.oschina.net/u/876354/blog/1787921)
Faster R-CNN可以简单地看做“区域生成网络+Fast R-CNN”的系统，Selective Search方法被区域生成网络(Region Proposal Network RPN)代替，最终使得候选区域生成、特征提取、分类、位置精修等四个基本步骤被同一到一个深度网络框架之内，运算无重复，速度大大提高。
![2018-11-22_142640](/assets/2018-11-22_142640.png)
根据上图，Faster RCNN可以分为如下四个主要内容：
* Conv layers：作为一种CNN网络目标检测方法，Faster RCNN首先使用一组基础的conv+relu+pooling层提取image的feature maps，该feature maps被共享用于后续RPN层和全连接层；
* Region Proposal Networks：RPN网络用于生成region proposals。该层通过softmax判断anchors属于foreground或者background，再利用bounding box regression修正anchors获得精确的proposals；
* RoI Pooling：该层收集输入的feature maps和proposals，综合这些信息后提取proposal feature maps，送入后续全连接层判定目标类别；
* Classification：利用proposal feature maps计算proposal的类别，同时再次bounding box regression获得检测框最终的精确位置；

下图展示了python版本中的VGG16模型中的faster_rcnn_test.pt的网络结构，是对上图的更为详细的展开：网络首先将图像缩放至MxN大小，然后经过Conv Layers形成Feature map；RPN网络对feature map处理生成Proposal(细节后面说)，ROI Pooling则利用Proposal信息从Feature map中提取出相应的proposal feature并送入全连接网络进行classification。
![2018-11-22_143702](/assets/2018-11-22_143702.png)

## RPN
RPN网络部分的结构如下图。前面提到，RPN网络的目的是生成region proposals，并通过softmax判断anchors属于foreground或者background，再利用bounding box regression修正anchors以获得精确的proposals，下面终点来解释anchors。
![2018-11-22_143702](/assets/2018-11-22_143702_w1m8lfaw1.png)
个人觉得作者提出的anchors思路简单粗暴。anchors可以理解为特征图中的一个特征点以及该点周围一定范围内的区域，作者将每一个特征点都作为一个anchor，并将该点及周围一定范围内的点送入softmax判断该区域是foreground还是background，而这个"一定范围"，作者在文中给出了9种不同的尺寸，如下图所示，绿、红、蓝各一组，每组三个，长宽比约为1:1/1:2/2:1三种，也即对每个点都要用这9种不同大小的框进行判断foreground和background。
![2018-11-22_151249](/assets/2018-11-22_151249.png)
下图更清晰地反应了这一思想，图中的k默认为9，也就是9种不同大小的框，假设网络输出的feature map为256维，由于对每一个anchor和k种不同大小的box，都要对其是foreground还是background进行评分，所以每一个anchor会有2k个scores，另外，每一个anchor和box都有[x,y,w,h]四个量存储其位置，所以每一个anchor会有4k个coordinates。
![2018-11-22_151642](/assets/2018-11-22_151642.png)
按照这种思路，如果生成的特征图是m*n的，那么最终得到的评分将会是m\*n\*2k个，事实上程序在训练时只会在复合要求的anchors中随机选取128个postive anchors+128个negative anchors进行训练。
再回到RPN网络结构部分，先进行1*1降维至18层？？
