---
layout:     post
title:      Deep Learning - Optimization
subtitle:   深度学习基础
date:       2019-04-03
author:     vhpg
header-img: img/placeholder_img.png
catalog: true
tags:
    - Machine Learning
    - Deep Learning
    - Optimization
    - Mathematics
---
> 本篇文章观点仅限于目前的理解，后续若有新的理解，还会继续更新。

> 文中编辑的公式，在网页上查看会有问题，可使用Atom+MarkDown插件查看(KaTex)

#### 0. Introduce
  本文对神经网络训练时常用的优化器进行总结，限于个人水平有限，无法从一个大的角度来谈深度学习中的优化问题，因此本文内容主要摘自[这篇文章](http://ruder.io/optimizing-gradient-descent/)，并加上一些自己的理解。

  总的来说，深度学习中的优化方法都是基于梯度下降的，都是对梯度下降方法的改进。所以本文首先介绍梯度下降算法及相应的变种，然后简单阐述在深度学习优化过程中的一些挑战，再介绍当前深度学习训练中一些流行的算法和其在Keras中的配置。


#### 1. Gradient descent and its variants
  梯度下降是一种最小化目标函数的方法。对于给定的目标函数$J(\theta)$，其中参数$\theta \in R^d$，我们可以通过求解目标函数关于参数$\theta$的梯度$\nabla_{\theta}J(\theta)$，并按照与梯度相反的方向更新$\theta$，通过逐步迭代更新即可找到目标函数的一个极小值点。

  根据在每次迭代更新参数$\theta$过程中所使用的样本数量的不同，梯度下降有以下三种不同的变体：

##### Batch gradient descent
  批梯度下降，该方法是每次在**所有样本**上计算损失函数关于参数的梯度，然后执行**一次更新**：
  $$\theta = \theta - \eta \cdot \nabla_{\theta}J(\theta)$$
  Batch gradient descent的伪代码形式为：
  ```
  for i in range(nb_epochs):
    params_grad = evaluate_gradient(loss_function, data, params)
    params = params - learning_rate * params_grad
  ```
  Batch gradient descent由于每次要遍历所有样本才执行一次更新，所以其收敛速度会非常慢，对于较大的无法全部加载到内存中的数据集不太好处理。另外，它无法及时利用新的数据在线更新模型。

##### Stochastic gradient descent
  Stochastic gradient descent(SGD)与Batch gradient descent正好相反，它是对每一个样本$(x^{(i)}, y^{(i)})$计算梯度然后立即执行一次更新：$$\theta = \theta - \eta \cdot \nabla_{\theta}J(\theta;x^{(i)}; y^{(i)})$$
  Stochastic gradient descent(SGD)的伪代码形式为：
  ```
  for i in range(nb_epochs):
    np.random.shuffle(data)
    for example in data:
      params_grad = evaluate_gradient(loss_function, example, params)
      params = params - learning_rate * params_grad
  ```
  在每一轮选取数据时，一般都要打乱数据，这是为了避免数据可能存在某种特别的顺序而导致参数向某一特定的方向移动。

  由于SGD对每一个样本都执行一次更新，每次更新都可能将损失函数带向不同的方向，所以SGD有在线利用新样本并有一定的跳出局部极小并转向更好的局部极小的能力。另一方面，如[Wikipedia](https://upload.wikimedia.org/wikipedia/commons/f/f3/Stogra.png)中的图片所示，对每个样本更新一次参数使得SGD处于一种不断超调的状态，最终会收敛到一个较好的局部极小，可以想象，只有到达这样的局部极小(或者说最小)，下次更新时参数才不会跳出来，才达到所谓的收敛。
  但在实际中发现，当我们不断降低学习率，SGD一般会收敛到与BGD相同的结果。
  ![2019-04-03_090229](/assets/2019-04-03_090229.png)

##### Mini-batch gradient descent
  Mini-batch gradient descent是结合了上述两种方法，对每$n$个样本的mini-batch执行一次更新：$$\theta = \theta - \eta \cdot \nabla_{\theta}J(\theta;x^{(i:i+n)}; y^{(i:i+n)})$$
  Mini-batch gradient descent的伪代码如下：
  ```
  for i in range(nb_epochs):
    np.random.shuffle(data)
    for batch in get_batches(data, batch_size=50):
      params_grad = evaluate_gradient(loss_function, batch, params)
      params = params - learning_rate * params_grad
  ```
  Mini-batch gradient descent可以减少像SGD那样的参数更新时的剧烈变化，使得更易达到稳定的收敛，并且在现有的深度学习库中，该方法可以充分利用高度优化的矩阵运算来高效地计算梯度。
  目前各种深度学习库中，使用的所谓的SGD方法其实都是该方法，本文后面要介绍的各个改进的梯度下降方法也是基于该方法。为了表达的简便，后面将省略$x^{(i:i+n)},y^{(i:i+n)}$。

#### 2. Challenges
  Mini-batch gradient descent方法有时并不能收敛到较好结果，并且还存在如下一些尚需解决的问题：

  * learning rate难以选择，太小时收敛慢，太大时会导致参数拨动而阻碍收敛，甚至可能导致发散；

  * 有一系列的learning rate schedules尝试在训练过程中调整学习率，如根据预定的策略，在训练一定的轮次后减小学习率，或当loss不再下降或验证集精度不再上升时减小学习率等等，但这些策略都是需要预先定义好，没法去适应数据集的特性；

  * 此外，我们的学习率是应用于所有参数，而不同的参数可能需要不同的学习率；

  * 最重要的一点是，对于一个高度非凸的目标函数，优化过程很可能陷入某一局部极小。除此之外，优化过程还有可能困于鞍点处(有的维度需要上升，有的维度需要下降)，主要是因为鞍点处通常会有一个平面，而在这个平面处计算得到的各个方向的梯度都接近为0，使得SGD很难跳出该平面。

#### 3. Gradient descent optimmization algorithms
  下面介绍的一些方法是在深度学习优化过程中使用到的一些方法，主要用于解决上面提到的一些挑战。
  这里首先贴上一个本文的主要参考博文的作者所写的一个各种优化方法对比的[脚本](http://louistiao.me/notes/visualizing-and-animating-optimization-algorithms-with-matplotlib/)，后续研究。
##### Momentum
  如下图所示，在这种一个方向明显需要比另一个方向下降快的情况，对于一般的SGD算法，通常会经过多次震荡，达到局部极小的速度过慢：
  ![2019-04-03_095733](/assets/2019-04-03_095733.png)
  而Momentum就是一种阻碍震荡并加速SGD在相应方向下降的方法，如下图所示：
  ![2019-04-03_100228](/assets/2019-04-03_100228.png)
  Momentum的思想是，每次更新参数时，不仅仅使用当前的梯度信息，也同时使用之前若干步的梯度信息，这样的好处是，如果在该方向上之前的梯度就很大(梯度一直指向一个方向)，那么当前梯度与之前梯度结合可大大加快下降速度，而对于那些梯度方向摇摆不定的参数，则会相应减弱其梯度的更新。具体公式如下：
  $$\begin{aligned}
    v_t &= \gamma v_{t-1} + \eta \nabla_{\theta}J(\theta) \\
    \theta &= \theta - v_t
  \end{aligned}$$
  公式中，$\gamma$的值一般设为0.9左右。

  想象一下我们把一个球扔下山，在有momentum的情况下，球下山的速度会越来越快，直到到达山底。要注意，$\gamma < 1$，所以在每次更新参数时，以前的梯度所占的比重会越来越小，也就是说，参数不会在到达局部极小时仍剧烈波动，但相比于没有momentum的情况，**有momentum时仍然会带来一定的跳出局部极小的能力，以及跳出上述challenge中所述的马鞍面的能力**。

##### Nesterov
  相比于momentum使球无脑地滚下山，Nesterov则相当于给球一点智慧，它会预先判断下一时刻可能到达的位置，从而可以在转向上坡前慢下来，相当于减小到达局部极小时的波动。

  在前面我们使用动量项$\gamma v_{t-1}$来参与指导参数的移动，那么我们考虑将$\theta - \gamma v_{t-1}$来作为下一时刻参数可能到达的位置的一个大致预测，我们对该预测参数求梯度，就相当于预判前面的梯度方向，若是与当前方向相同，则会继续加速向前，若是与该方向相反，则会抵消当前的一部分速度，使得下降减速，具体公式如下：
  $$\begin{aligned}
    v_t &= \gamma v_{t-1} + \eta \nabla_{\theta}J(\theta - \gamma v_{t-1}) \\
    \theta &= \theta - v_t
  \end{aligned}$$

  引用[这里](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)的一张示意图来对Momentum和Nesterov方法进行解释：
  图中短的蓝色箭头相当于初始的梯度下降方向和相应值，而长的蓝色箭头则相当于再一次迭代时由于momentum和当前梯度联合所导致的梯度下降方向和值；
  图中的棕色箭头表示原有动量项所导致的梯度下降方向和相应值，而红色箭头则是对下一时刻梯度方向和值的预测，两向量相加后得到的绿色箭头则是修正后的梯度下降方向和值，也即Nesterov方法所带来的梯度方向；
  ![2019-04-03_103451](/assets/2019-04-03_103451.png)

  上述两种方法在一定程度上解决了challenge中的第四点，在大多数深度学习库中，SGD方法也都包含上述两种方法的参数配置，以下是Keras中的相应配置：
  ```
    keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    Arguments:
    * lr: Learning rate
    * momentum: 动量项参数，也即公式中的gamma，一般设置为0.9左右
    * decay: 每次更新参数时的lr衰减系数
    * nesterov: 是否使用nesterov
  ```

##### Adagrad
  在之前的方法中，所有的参数$\theta_i$共同使用一个学习率$\eta$，而Adagrad则在第$t$时刻时为每个参数$\theta_i$使用不同的学习率。

  为了表示简介，记$g_t$为第$t$时刻时的梯度信息，$g_{t, i}$表示在第$t$时刻时目标函数关于参数$\theta_i$的梯度信息，即: $g_{t,i} = \nabla_{\theta}J(\theta_{t,i})$，则一般的SGD方法在$t$时刻关于参数$\theta_i$的更新公式为：
  $$\theta_{t+1, i} = \theta_{t, i} - \eta \cdot g_{t, i}$$

  在Adagrad方法中，该公式被更改为：
  $$\theta_{t+1, i} = \theta_{t, i} - \frac{\eta}{\sqrt{G_{t,ii}+\varepsilon}} \cdot g_{t, i}$$
  式中，$G_t \in \Bbb{R}^{d*d}$，为一个对角阵，对角上每一个元素为相应元素历史梯度的平方和，$\varepsilon$为一个平滑项，避免分母为0，通常设为$1e-8$。这样的配置就相当于，对于梯度一直非常大的参数，其相应的学习率就会减小，而对于梯度非常小的参数，其相应的学习率就会稍大。另外要注意，上述公式中若没有开方，实际效果就会差很多。

  上述公式使用向量化的表示为：
  $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t+\varepsilon}} \odot g_t$$

  Adagrad的优点是，它通常不需要人为地设定learning rate的调整策略，大多数情况下只需设定$lr=0.01$即可；
  但是Adagrad的缺点也非常明显，由于分母中的$G_t$一直是正向累加，所以会越来越大，最终会导致学习率非常小以至于参数不再继续更新。

  Keras中Adagrad方法的配置如下：
  ```
    keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
    Arguments:
    * lr: Learning rate
    * epsilon: 即公式中的epsilon参数，若为None，则使用默认的epsilon参数值
    * decay: 每次更新参数时的lr衰减系数
  ```

##### Adadelta
  Adadelta的目的是为了解决Adagrad方法中learning rate连续不断下降的问题。相比于Adagrad记录所有历史梯度的平方和，Adadelta则是使用一种递归的方式记录历史梯度的平方和：当前梯度平方和的均值等于历史梯度平方和均值和当前梯度平方和的加权和。公式如下：
  $$E[g^2]_t = \gamma E[g^2]_{t-1} + (1-\gamma)g_t^2$$
  上式中的$E[g^2]_t$就相当于Adagrad中的$G_t$，这样定义的好处是分母不会无限增大，当某些参数的梯度逐渐减小时，$E[g^2]_t$也会随着迭代次数的增加而减小，进而使得对这些参数的学习率增大。

  对$\Delta \theta_t = -\frac{\eta}{\sqrt{G_t+\varepsilon}}g_t = -\frac{\eta}{\sqrt{E[g^2]_t+\varepsilon}}g_t=-\frac{\eta}{RMS[g]_t}g_t$，($RMS[g]_t$表示梯度的平方根)，作者表示该式中等号前后量纲不一致，(前面各种方法中量纲其实都不一致)，为了使量纲一致，作者又对该参数更新量求一次平方和均值，公式如下：
  $$E[\Delta \theta^2]_t = \gamma E[\Delta \theta^2]_{t-1} + (1-\gamma)\Delta \theta_t^2$$
  该式可简写为:$RMS[\Delta \theta]_t = \sqrt{E[\Delta \theta^2]_t + \varepsilon}$
  而式中$RMS[\Delta \theta]_t$在计算时为未知，可以使用$RMS[\Delta \theta]_{t-1}$代替。
  最终的参数更新公式为：
  $$\begin{aligned}
    \Delta \theta_t &= -\frac{RMS[\Delta \theta]_{t-1}}{RMS[g]_t}g_t \\
    \theta_{t+1} &= \theta_t + \Delta \theta_t
  \end{aligned}$$
  上述公式表明，使用Adadelta连基本的学习率都不用设置。

  Keras中Adadelta方法并没有使用上述最原始的更新方法，而是提供了一种带有学习率的方法，具体配置如下：
  ```
    keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    Arguments
    * lr: 默认为1，建议保持默认值
    * rho: 即公式中的 gamma 参数
    * epsilon: 即公式中的 epsilon 参数
    * decay: lr衰减系数
  ```

##### RMSprop
  RMSprop方法与Adadelta类似，也是为了解决Adagrad中学习率不断下降的问题。实际上，RMSprop就相当于$\gamma=0.9$时的Adadelta，公式如下：
  $$\begin{aligned}
      E[g^2]_t &= 0.9E[g^2]_{t-1} + 0.1g_t^2 \\
      \theta_{t+1} &= \theta_t -\frac{\eta}{\sqrt{E[g^2]_t+\varepsilon}}g_t
    \end{aligned}$$
  RMSprop在RNN训练中通常是不错的选择。
  Keras中RMSprop方法的配置如下：
  ```
    keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    Arguments
    * lr: 建议保留默认值
    * rho: gamma参数，建议默认值
    * epsilon: epsilon参数
    * decay: lr衰减系数
  ```

##### Adam (almost)
  为每个参数配置相应的学习率，Adagrad/Adadelta/RMSprop算是一类方法，而Adam算是另一类方法。
  Adam不仅像Adadelta和RMSprop一样存储了历史梯度的指数衰减的平方均值，还存储了历史梯度的指数衰减的均值，该项相当于动量项，具公式如下：
  $$\begin{aligned}
      m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \\
      v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
    \end{aligned}$$
  下面这一点自己不是很理解：
  式中，$m_t$和$v_t$相当于是对梯度的均值和方差的估计，由于这两个向量都使用0进行初始化，作者观察到在初始训练阶段以及当衰减指数$\beta_1$和$\beta_2$接近1时，$m_t$和$v_t$都相对于0有偏差。作者通过计算以下计算来抵消这些偏差：
  $$\begin{aligned}
      \hat{m}_t = \frac{m_t}{1-\beta_1^t} \\
      \hat{v}_t = \frac{v_t}{1-\beta_2^t}
    \end{aligned}$$
  最终的更新规则为：
  $$\theta_{t+1}=\theta_t - \frac{\eta}{\sqrt{\hat{v}_t}+\varepsilon}\hat{m}_t$$
  作者建议的默认参数为:$\beta_1=0.9$，$\beta_2=0.999$，$\varepsilon=1e-8$。
  作者通过实验表明了该方法在实际应用中表现较好，优于其他同类方法。

##### AMSGrad
  适应学习率的方法在神经网络训练中经常被使用，但一些人发现其在物体识别或机器翻译等项目中表现并不好，不能收敛到最优解，并且性能不如SGD+momentum。有学者指出，其性能不好的原因主要是对历史梯度的平方进行了指数衰减的平均，而当时这么这么做的原因主要是解决Adagrad中学习率不断下降的问题。

  通过实验发现，Adam收敛到局部极小的原因是，只有一些batch提供了较大的且包含信息梯度，而这些batch出现的次数又很少，所以这些梯度信息总是在不断地迭代更新中逐渐消失，因而导致了较差的收敛性。

  AMSGrad就是为了解决这个问题，作者在求$\hat{v}_t$时，改用最大值的方法，并且去除了Adam中求解$\hat{m}_t$的步骤，具体公式如下：
  $$\begin{aligned}
      m_t &= \beta_1 m_{t-1} + (1-\beta_1)g_t \\
      v_t &= \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \\
      \hat{v}_t &= max(\hat{v}_{t-1}, v_t) \\
      \theta_{t+1}&=\theta_t - \frac{\eta}{\sqrt{\hat{v}_t}+\varepsilon}m_t
    \end{aligned}$$

  目前的实验是，AMSGrad方法在小数据集上效果优于Adam，而在其他一些实验中效果并不比Adam好，具体还需继续观察。

  Keras中并没有专门的AMSGrad的优化函数，而是将其和Adam写在一起，Adam方法的配置如下，建议使用默认配置，另外需要注意，这一类适应学习率的方法应该都不怎么需要decay参数：
  ```
    keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    Arguments
    * amsgrad: 是否使用AMSGrad方法对参数进行更新
  ```

##### AdaMax
  AdaMax方法是将梯度公式中的$l_2$模换成了$l_{\infin}$模，改进者认为这样可使$v_t$收敛到更为稳定的值，为了避免混淆，这里使用$u_t$进行表示，公式如下：
  $$u_t = \beta_2^{\infin} v_{t-1} +(1-\beta_2^{\infin})|g_t|^{\infin}=max(\beta_2 \cdot v_{t-1}, |g_t|)$$
  最终的AdaMax的更新规则如下：
  $$\theta_{t+1} = \theta_t - \frac{\eta}{u_t}\hat{m}_t$$

  Keras中AdaMax的配置如下，建议都使用默认配置：
  ```
    keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
  ```

##### Nadam
  从前面的分析可以看出，Adam方法相当于是结合了RMSprop和Momentum，而Nesterov又是对Momentum的进一步优化，所以这里的Nadam相当于是RMSprop和Nestrov。
  这里没看太懂，暂时不写。
  Nadam最终的更新公式为：
  $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t}+\varepsilon}(\beta_1\hat{m}_t+\frac{(1-\beta_1)g_t}{1-\beta_1^t})$$

  Keras中Nadam方法的参数配置如下，建议保留默认值：
  ```
    keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
  ```

#### 4. Which optimizer to use
  若输入数据为稀疏的，适应学习率的方法一般能给出较好的结果。
  一句话，Adam是目前的最好选择。

#### Reference
[Blog](http://ruder.io/optimizing-gradient-descent/)

[Keras](https://keras.io/optimizers/)
