# Some definitions about machine learning in 19-22 July


- ## 机器学习中的维度灾难 (The Curse of Dimensionality in classification)
> ref: 
> - [Translation version](https://blog.csdn.net/red_stone1/article/details/71692444)
> - [Original version](https://www.visiondummy.com/2014/04/curse-dimensionality-affect-classification/)

---

- ## Cross entropy
> ref: https://en.wikipedia.org/wiki/Cross_entropy

Related to: Kullback–Leibler divergence (K-L散度)
> ref: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

KL divergence definition: how one probability distribution P is different from a second, reference probability distribution Q.

> relating reading resources:
> - https://zhuanlan.zhihu.com/p/54066141
> - https://zhuanlan.zhihu.com/p/149186719

    

**Cross Entropy**：$H(p,q) = -\sum\limits_{x \in \chi} p(x) \log q(x)$ ，一般是用来量化两个概率分布之间差异的损失函数（多用于分类问题）。

计算出Cross Entropy Loss之后，用Gradient Descent来进行参数的更新。


> ref: [简单谈谈Cross Entropy Loss](https://blog.csdn.net/xg123321123/article/details/80781611)

Q: 为什么机器学习引入了Cross Entropy Loss?  
Ans: 分类问题和回归问题是监督学习的两大种类：**分类**问题的目标变量是**离散**的；回归问题的目标变量是连续的数值。神经网络模型的效果及优化的目标是通过损失函数来定义的。

回归问题解决的是对具体数值的预测。比如房价预测、销量预测等都是回归问题。这些问题需要预测的不是一个事先定义好的类别，而是一个任意实数。对于**回归**问题，常用的损失函数是**均方误差( MSE，mean squared error )。**

**分类**问题常用的损失函数为**交叉熵( Cross Entropy Loss)**。交叉熵描述了两个概率分布之间的距离，当交叉熵越小说明二者之间越接近。尽管交叉熵刻画的是两个概率分布之间的距离，但是神经网络的输出却不一定是一个概率分布。为此我们常常用**Softmax回归**将神经网络前向传播得到的结果变成概率分布。

$softmax$常用于多分类过程中，它将多个神经元的输出，归一化到 $(0, 1)$ 区间内，因此Softmax的输出可以看成概率，从而来进行多分类。一个包含 $k$ 个元素的数组 $V$，$i$ 表示 $V$ 中的第 $i$ 个元素，那么这 $i$ 个元素的 $softmax$ 输出就是:
$$
S_i = \frac{e^i}{\sum_{j = 1}^k e^j}
$$

<br/>
<br/>

> ref: [Softmax 和 Softmax-loss的推演](https://blog.csdn.net/xg123321123/article/details/52716890)

> refref: [Softmax vs. Softmax-Loss: Numerical Stability](https://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability/) (包含了Back Propagation的内容)

Softmax 函数在 Logistic Regression 里的作用是将线性预测值转化为类别概率。
现在每个 $S_i$ 就可以解释成观察到的数据 x 属于类别 i 的概率，或者称作似然 (Likelihood)。

最大似然就是要最大化 $S_i$ 的值 (通常是使用 negative log-likelihood 而不是 likelihood，也就是说最小化 $−log(S_i)$ 的值，这两者结果在数学上是等价的)。(以下，用 $O_y$ 代替 $S_i$ )

Softmax-Loss 就是把两者结合到一起:
$$
\mathcal{L}(y,o) = -\log (\frac{e^{o_y}}{\Sigma_{j=1}^m e^{o_j}}) = \log (\Sigma_{j=1}^m e^{o_j}) - o_y
$$
目标是最小化上面这个等式。比较自然地，将softmax和loss合在了一起来考虑，或者甚至还可以将 $z_i=w^T_ix+b_i$ 的定义也一起带进去，然后对w和b进行求导来得到梯度下降的链式法则。


<br/>
<br/>
<br/>

**Classification-Erroe**
$$
classification-error = \frac{count-of-error-items}{count-of-all-items}
$$

分类问题用 One Hot Label + Cross Entropy Loss
1. Training 过程，分类问题用 Cross Entropy Loss，回归问题用 Mean Squared Error。
2. validation / testing 过程，使用 Classification Error更直观，也正是我们最为关注的指标。


Q: 为什么回归问题用 MSE?  
Ans: 最小二乘是在欧氏距离为误差度量的情况下，由系数矩阵所张成的向量空间内对于观测向量的最佳逼近点。



Q: 为什么用欧式距离作为误差度量 （即 MSE）？  
Ans: 
1. 它简单。

2. 它提供了具有很好性质的相似度的度量。

    1）它是非负的;

    2）唯一确定性。只有 x=y 的时候，d(x,y)=0；

    3）它是对称的，即 d(x,y)=d(y,x)；

    4）符合三角性质。即 d(x,z)<=d(x,y)+d(y,z).

3. 物理性质明确，在不同的表示域变换后特性不变，例如帕萨瓦尔等式。

4. 便于计算。通常所推导得到的问题是凸问题，具有对称性，可导性。通常具有解析解，此外便于通过迭代的方式求解。

5. 和统计和估计理论具有关联。在某些假设下，统计意义上是最优的。


> ref: IEEE Signal Processing Magzine 的 《Mean squared error: Love it or leave it?》

MSE 的缺点:
1. 信号的保真度和该信号的空间和时间顺序无关。即，以同样的方法，改变两个待比较的信号本身的空间或时间排列，它们之间的误差不变。例如，[1 2 3], [3 4 5] 两组信号的 MSE 和 [3 2 1],[5 4 3] 的 MSE 一样。

2. 误差信号和原信号无关。只要误差信号不变，无论原信号如何，MSE 均不变。例如，对于固定误差 [1 1 1]，无论加在 [1 2 3] 产生 [2 3 4] 还是加在 [0 0 0] 产生 [1 1 1]，MSE 的计算结果不变。

3. 信号的保真度和误差的符号无关。即对于信号 [0 0 0]，与之相比较的两个信号 [1 2 3] 和[-1 -2 -3] 被认为和 [0 0 0] 具有同样的差别。

4. 信号的不同采样点对于信号的保真度具有同样的重要性。

<br/>

### 熵(Entropy)大合集 
<br/>

> ref: [熵](https://blog.csdn.net/rtygbwwwerr/article/details/50778098)

<br/>

**1. 信息量**：$I(x_0) = -\log (p(x_0))$，$p(x_0)$是随机变量的概率分布函数。一个事件发生的概率越大，则它所携带的信息量就越小。

<br/>

**2. 熵**：对于一个随机变量X而言，它的所有可能取值的信息量的期望（$E[I(x)]$）就称为熵。$$H(x) = E_p \log \frac{1}{p(x)} = - \sum_{x \in \chi} p(x) \log p(x)$$ 如果p(x)是连续型随机变量的pdf，则熵定义为：
$$H(x) = - \int \limits_{x \in \chi} p(x) \log p(x) dx$$

为了保证有效性，这里约定当 $p(x) \rightarrow 0$ 时,有 $p(x)logp(x) \rightarrow 0$，当两种取值的可能性相等时，不确定度最大。当 $p = 0$ 或 $1$时，熵为 $0$，即此时X完全确定为 $0$。

Note: 熵的单位随着公式中 $\log$ 运算的底数而变化，当底数为2时，单位为“比特”(bit)，底数为e时，单位为“奈特”。

<br/>

**3. 条件熵**: 在随机变量X发生的前提下，随机变量Y发生所新带来的熵定义为Y的条件熵，用表示，用来衡量在已知随机变量X的条件下随机变量Y的不确定性。

$$
H(Y|X)=H(X,Y)–H(X)
$$

数学推导如下：

$$
H(X,Y)–H(X) = -\sum_{x,y} p(x,y) \log p(x,y) + \sum_{x} p(x) \log p(x)
$$

$$
= -\sum_{x,y} p(x,y) \log p(x,y) + \sum_{x} \left( \sum_{y} p(x,y)   \right) \log p(x)
$$

$$
= -\sum_{x,y} p(x,y) \log p(x,y) + \sum_{x,y} p(x,y) \log p(x)
$$

$$
= -\sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)}
$$

$$
= -\sum_{x,y} p(x,y) \log p(y | x) = H(y | x)
$$


<br/>

**4. 相对熵**: 相对熵(relative entropy)又称为 $KL$ 散度（Kullback-Leibler divergence）。KL距离，是两个随机分布间距离的度量。
记为$DKL(p||q)$。它度量*当真实分布为p时，假设分布q的无效性*。

$$
D_{KL}(p||q) = E_p[\log \frac{p(x)}{q(x)}] = \sum \limits_{x \in \chi} p(x) \log \frac{p(x)}{q(x)} 
$$

$$
= \sum \limits_{x \in \chi} p(x) \log p(x) - \sum \limits_{x \in \chi} p(x) \log q(x) 
$$

$$
= -H(p) - \sum \limits_{x \in \chi} p(x) \log q(x) 
$$

$$
= -H(p) + E_p[-\log q(x)] = H_P(q) - H(p)
$$

$H_p(q)$ 表示在p分布下，使用 $q$ 进行编码需要的bit数。当 $p=q$ 时,两者之间的相对熵 $D_{KL}(p||q) = 0$。$D_{KL}(p||q)$ 表示*在真实分布为p的前提下*，使用q分布进行编码相对于使用真实分布p进行编码（即最优编码）所**多**出来的bit数。

<br/>

**5. 交叉熵**: 假设有两个分布 $p, q$，则它们在给定样本集上的交叉熵定义如下：

$$
CEH(p,q) = E_p[-\log q] = - \sum \limits_{x \in \Chi} p(x) \log q(x) = H(p) + D_{KL}(p || q)
$$

交叉熵与上一节定义的相对熵仅相差了 $H(p)$，当p已知时，可以把 $H(p)$ 看做一个常数，此时交叉熵与 $KL$ 距离在行为上是等价的，都反映了分布 $p，q$ 的相似程度。最小化交叉熵等于最小化 $KL$ 距离。它们都将在 $p=q$ 时取得最小值 $H(p)$ $(p=q时KL距离为0)$

特别的，在logistic regression中，   
p:真实样本分布，服从参数为p的0-1分布，即$X∼B(1,p)$   
q:待估计的模型，服从参数为q的0-1分布，即$X∼B(1,q)$
$$
CEH(p,q) = -\sum_{x \in \Chi} p(x) \log q(x) 
$$

$$
= - [P_p(x = 1) \log P_q (x = 1) + P_p(x = 0) \log P_q (x = 0)]
$$

$$
= - [p\log q + (1-p)\log (1-q)]
$$

$$
= - [y \log h_\theta (x) + (1-y) \log (1 - h_\theta (x))]
$$

对所有的样本取均值可以获得
$$
CHE = - \frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log h_\theta (x^{(i)}) + (1 - y^{(i)}) (1 - \log h_\theta (x^{(i)}))]
$$ 
这个结果与通过最大似然估计方法求出来的结果一致。

<br/>

> 补充 ref: [熵与信息增益](https://blog.csdn.net/xg123321123/article/details/52864830)

> 补充 [softmax交叉熵损失函数求导](https://blog.csdn.net/qian99/article/details/78046329) (求导的链式法则)


**6. 互信息**: 
两个随机变量 $X，Y$ 的互信息定义为 $X，Y$ 的联合分布和各自独立分布乘积的相对熵，用 $I(X,Y)$ 表示

$$
I(x,y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x) p(y)}
$$

熵与条件熵之差称为互信息，
$H(y) - I(x,y) = H(y|x)$ 或者 $H(x) - I(x,y) = H(x|y)$。
推导如下：

$$
H(Y) - I(x,y) = -\sum_y p(y) \log p(y) - \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x) p(y)}
$$

$$
= -\sum_y \left( \sum_x p(x,y) \right) \log p(y) - \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x) p(y)}
$$

$$
= -\sum_{x,y} p(x,y) \log p(y) - \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x) p(y)}
$$

$$
= -\sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)} 
$$

$$
= -\sum_{x,y} p(x,y) \log p(y | x) = H(y | x)
$$

7. **信息增益**

在决策树ID3算法中，使用信息增益来选择最佳的特征作为决策点。

信息增益表示得知特征X的信息而使得类Y的信息不确定性减少的程度，即用来衡量特征X区分数据集的能力。当新增一个属性X时，信息熵的变化大小即为信息增益。 越大表示X越重要。

$$
I(Y | X) = H(Y) - H(Y | X)
$$



<br/>

---

- ## One-hot Encoding

> ref: 数据预处理：[独热编码（One-Hot Encoding）和 LabelEncoder标签编码](https://www.cnblogs.com/zongfa/p/9305657.html) (Python Pandas库的实现)
>> 先验阅读材料: [softmax交叉熵损失函数求导](https://blog.csdn.net/qian99/article/details/78046329)

<br/>

独热编码即 One-Hot 编码，又称一位有效编码，其方法是使用N位状态寄存器来对N个状态进行编码，每个状态都由他独立的寄存器位，并且在任意时候，其中只有一位有效。

这样做的好处主要有：
1. 解决了分类器不好处理属性数据的问题
2. 在一定程度上也起到了扩充特征的作用

为什么要独热编码？因为大部分算法是基于向量空间中的度量来进行计算的，为了使非偏序关系的变量取值不具有偏序性，并且到圆点是等距的。使用 $one-hot$ 编码，将离散特征的取值扩展到了欧式空间，离散特征的某个取值就对应欧式空间的某个点。将离散型特征使用one-hot编码，会让特征之间的距离计算更加合理。离散特征进行one-hot编码后，编码后的特征，其实每一维度的特征都可以看做是连续的特征。就可以跟对连续型特征的归一化方法一样，对每一维特征进行归一化。比如归一化到 $[-1,1]$ 或归一化到均值为0,方差为1。       

缺点：当类别的数量很多时，特征空间会变得非常大。在这种情况下，一般可以用PCA来减少维度。而且one hot encoding+PCA这种组合在实际中也非常有用。

<br/>

Q: 什么是PCA?   <!--//todo（暂未阅读）-->   
Ans: Principal Component Analysis，是一种常见的数据分析方式，常用于高维数据的降维，可用于提取数据的主要特征分量。
> ref: [【机器学习】降维——PCA](https://zhuanlan.zhihu.com/p/77151308)


什么情况下(不)用独热编码？
- 用：用来解决类别型数据的离散值问题，

- 不用：将离散型特征进行one-hot编码的作用，是为了让距离计算更合理，但如果特征是离散的，并且不用one-hot编码就可以很合理的计算出距离，那么就没必要进行one-hot编码。

<br/>

什么情况下(不)需要归一化？
- 需要： 基于参数的模型或基于距离的模型，都是要进行特征的归一化。
- 不需要：基于树的方法是不需要进行特征的归一化，例如随机森林，bagging 和 boosting等。



---

- ## Transfer learning & Transformer (machine learning model)

> [Transfer learning Wikipedia](https://en.wikipedia.org/wiki/Transfer_learning)    
> [Transformer Wikipedia](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model))


1. Transfer learning (TL) is a research problem in machine learning (ML) that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem.


Transfer Learning的初衷是节省人工标注样本的时间，让模型可以通过已有的标记数据（source domain data）向未标记数据（target domain data）迁移。从而训练出适用于target domain的模型。


2. A transformer is a deep learning model that adopts the mechanism of self-attention, differentially weighting the significance of each part of the input data. It is used primarily in the fields of natural language processing (NLP) and computer vision (CV).

Transformer VS RNN   
Like recurrent neural networks (RNNs), transformers are designed to process sequential input data, such as natural language, with applications towards tasks such as translation and text summarization. However, unlike RNNs, transformers process the entire input all at once. The attention mechanism provides context for any position in the input sequence.


> ref: [十分钟理解transformer](https://zhuanlan.zhihu.com/p/82312421)

当输入一个文本的时候，该文本数据会先经过一个叫Encoders的模块，对该文本进行编码，然后将编码后的数据再传入一个叫Decoders的模块进行解码，解码后就得到了翻译后的文本，对应的我们称Encoders为编码器，Decoders为解码器。

编码模块里边，有很多小的编码器，一般情况下，Encoders里边有6个小编码器，同样的，Decoders里边有6个小解码器。

在编码部分，每一个的小编码器的输入是前一个小编码器的输出，而每一个小解码器的输入不光是它的前一个解码器的输出，还包括了整个编码部分的输出。

encoder里边的结构是一个自注意力机制 (Self-attention) 加上一个前馈神经网络



---

- ## Data augmentation

> ref: [Wikipedia](https://en.wikipedia.org/wiki/Data_augmentation)

Data augmentation in data analysis are techniques used to increase the amount of data by adding slightly modified copies of already existing data or newly created synthetic data from existing data. It acts as a regularizer and helps reduce overfitting when training a machine learning model.


> An essay about the data agumentation: [A survey on Image Data Augmentation for Deep Learning](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0)

**Abstract**  
Deep convolutional neural networks have performed remarkably well on many Computer Vision tasks. However, these networks are heavily reliant on big data to avoid overfitting. Overfitting refers to the phenomenon when a network learns a function with very high variance such as to perfectly model the training data. Unfortunately, many application domains do not have access to big data, such as medical image analysis. This survey focuses on Data Augmentation, a data-space solution to the problem of limited data. Data Augmentation encompasses a suite of techniques that enhance the size and quality of training datasets such that better Deep Learning models can be built using them. The image augmentation algorithms discussed in this survey include geometric transformations, color space augmentations, kernel filters, mixing images, random erasing, feature space augmentation, adversarial training, generative adversarial networks, neural style transfer, and meta-learning. The application of augmentation methods based on GANs are heavily covered in this survey. In addition to augmentation techniques, this paper will briefly discuss other characteristics of Data Augmentation such as test-time augmentation, resolution impact, final dataset size, and curriculum learning. This survey will present existing methods for Data Augmentation, promising developments, and meta-level decisions for implementing Data Augmentation. Readers will understand how Data Augmentation can improve the performance of their models and expand limited datasets to take advantage of the capabilities of big data.



---

- ## Learning rate

> [wikipedia](https://en.wikipedia.org/wiki/Learning_rate)

Learning rate is a tuning parameter (Hyperparameter) in an optimization algorithm that determines the step size at each iteration while moving toward a minimum of a loss function

It metaphorically represents the speed at which a machine learning model "learns"

学习率越低，损失函数的变化速度就越慢，容易过拟合。虽然使用低学习率可以确保我们不会错过任何局部极小值，但也意味着我们将花费更长的时间来进行收敛，特别是在被困在局部最优点的时候。而学习率过高容易发生梯度爆炸，loss振动幅度较大，模型难以收敛。

<br/>

**如何设置初始学习率?**

首先设置一个十分小的学习率，在每个epoch之后增大学习率，并记录好每个epoch的loss或者acc，迭代的epoch越多，那被检验的学习率就越多，最后将不同学习率对应的loss或acc进行对比。

> ref: [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)

---

- ## Linear model

> [Wiki definition](https://en.wikipedia.org/wiki/Linear_model)  
> ref mathwork: [Linear Model -- Mathworks](https://www.mathworks.com/discovery/linear-model.html)

Definition: Describe mathematical relationships and make predictions from experimental data. Describe a continuous response variable as a function of one or more predictor variables. They can help you understand and predict the behavior of complex systems or analyze experimental, financial, and biological data.

The general equation for a linear model is:
$$
y = \beta_0 + \sum \beta_i X_i + \epsilon_i
$$

where  β  represents linear parameter estimates to be computed and  ϵ  represents the error terms.

There are several types of linear regression:
- **Simple linear regression**: models using only one predictor
- **Multiple linear regression**: models using multiple predictors
- **Multivariate linear regression**: models for multiple response variables




- ### 线性模型
线性模型和非线性模型的区别并不在于能不能去拟合曲线。

在统计意义上，如果一个回归等式是线性的，那么它的相对于参数就必须也是线性的。如果相对于参数是线性，那么即使性对于样本变量的特征是二次方或者多次方，这个回归模型也是线性的：$y = \omega_0 + \omega_1 x_1 + \omega_2 x_2^2$


甚至可以使用 log 或者指数去形式化特征：$y = \omega_0 + \omega_1 exp(-x_1) + \omega_2 exp(-x_2^2)$

- ### 非线性模型
参数是不是非线性的，非线性有很多种形象，因此可以较好的你和曲折的函数曲线，比如：$y = \theta_1 \times x^{\theta_2}$, $y = \theta_1 + (\theta_2 - \theta_3) \times exp(-\theta_4 x)$，这些非线性模型的特征因子对应的参数不止一个。

**Note:**
1. 线性模型可以是用曲线拟合样本，但是分类的决策边界一定是直线的，例如 $logistic$ 模型；
2. 区分是否为线性模型，主要是看一个乘法式子中自变量 $x$ 前的系数 $\omega$ ，应该是说 $x$ 只被一个 $\omega$ 影响，那么此模型为线性模型。或者判断决策边界是否是线性的；
3. 最简单判别一个模型是否为线性的，只需要判别*决策边界是否是直线*，也就是是否能用一条直线来划分


<br/>

- ### Logistic regression
> 三节相关的网课（台大李宏毅）
>> [Backpropagation](https://youtu.be/ibJpTrp5mcE)   
>> [Logistic Regression](https://youtu.be/hSXFuypLukA)  
>> [Classification](https://youtu.be/fZAZUYEeIMg)  

> Ref: [Wiki definition](https://en.wikipedia.org/wiki/Logistic_regression)   
> Ref: [IBM ref](https://www.ibm.com/topics/logistic-regression)

This type of statistical model (also known as logit model) is often used for classification and predictive analytics.
Logistic regression estimates the probability of an event occurring

逻辑回归（Logistic Regression）主要解决二**分类**问题，用来表示某件事情发生的可能性。

<br/>

逻辑回归 vs 线性回归

逻辑回归（Logistic Regression）与线性回归（Linear Regression）都是一种广义线性模型（generalized linear model）。逻辑回归假设因变量 y 服从伯努利分布，而线性回归假设因变量 y 服从高斯分布。 因此与线性回归有很多相同之处，去除 $Sigmoid$ 映射函数的话，逻辑回归算法就是一个线性回归。可以说，逻辑回归是以线性回归为理论支持的，但是逻辑回归通过 $Sigmoid$ 函数引入了非线性因素，因此可以轻松处理 $0/1$ 分类问题。

假设函数（Hypothesis function）

<!-- 后续还有ReLu函数 -->
$Sigmoid$ 函数，也称为逻辑函数（Logistic function): $g(z) = \frac{1}{1 + e^{-z}}$ (其中的$z = \sum_i w_i \cdot x_i + b$)，它的取值在[0, 1]之间，在远离0的地方函数的值会很快接近0或者1。

<br/>

- 决策边界（Decision Boundary）

也称为决策面，是用于在N维空间，将不同类别样本分开的平面或曲面。

**Note:** 决策边界是假设函数的属性，由参数决定，而不是由数据集的特征决定。决策边界其实就是一个方程

分类
- 线性决策边界
- 非线性决策边界



---

- ## Decision tree



---

- ## Self-Supervised Learning Vs Semi-Supervised Learning

As we work with bigger models, it becomes difficult to label all the data. Additionally, there is just not enough labelled data for a few tasks, such as training translation systems for low-resource languages. 



- Self-supervised learning (SSL)

This technique obtains a supervisory signal from the data by leveraging the underlying structure. The general method for self-supervised learning is to predict unobserved or hidden part of the input. 

Self-supervised learning 是无监督学习里面的一种，主要是希望能够学习到一种通用的特征表达用于下游任务。 其主要的方式就是通过自己监督自己，比如把一段话里面的几个单词去掉，用他的上下文去预测缺失的单词，或者将图片的一些部分去掉，依赖其周围的信息去预测缺失的 patch。

> ref: [自監督學習 SELF-SUPERVISED LEARNING 介紹](https://jigfopsda.com/zh/posts/2021/self_supervised_learning/)

> ref: [Self-Supervised Learning](https://project.inria.fr/paiss/files/2018/07/zisserman-self-supervised.pdf)


- Semi-Supervised Learning

Semi-supervised learning is a combination of supervised and unsupervised learning. It uses a small amount of labelled data with a larger share of unlabelled data. 

> ref: [Self-Supervised Learning Vs Semi-Supervised Learning: How They Differ](https://analyticsindiamag.com/self-supervised-learning-vs-semi-supervised-learning-how-they-differ/)

>ref: [Top 8 Resources To Learn Self-Supervised Learning In 2021](https://analyticsindiamag.com/top-8-resources-to-learn-self-supervised-learning-in-2021/)


---

- 怎样更好地理解并记忆泰勒展开式？ 
> ref: https://www.zhihu.com/question/25627482/answer/313088784

- 如何理解傅里叶变换公式？
> ref: https://www.zhihu.com/question/19714540/answer/514107420

傅里叶变换认为一个周期函数(信号)包含多个频率分量，任意函数（信号）f(t)可通过多个周期函数（基函数）相加而合成。

物理角度理解傅里叶变换是以一组特殊的函数（三角函数）为正交基，对原函数进行线性变换，物理意义便是原函数在各组基函数的投影。

- 一些有趣的数学推导
> [陈二喜的回答合集](https://zhuanlan.zhihu.com/p/95179405)


---

- ## 人工智能学习的理论教程集合
> [AI-EDU 微软人工智能教育与学习共建社区](https://microsoft.github.io/ai-edu/index.html)

微软人工智能教育与学习共建社区（Microsoft AI Education Community, 简称AI-Edu）是微软亚洲研究院（Microsoft Research Asia，简称MSRA）人工智能教育团队创立的人工智能开源社区。

本社区由基础教程、实践案例、实践项目三大模块构成，通过系统化的理论教程和丰富多样的实践案例，帮助学习者学习并掌握人工智能的知识，并锻炼在实际项目中的开发能力。


---

- ## Latent Dirichlet Allocation (LDA) in NLP

> ref: [潜在狄利克雷分布（LDA）初探](https://blog.csdn.net/VariableX/article/details/106385012)

潜在狄利克雷分布（Latent Dirichlet Allocation, LDA），是一种无监督学习算法，用于识别文档集中潜在的主题词信息。在训练时不需要手工标注的训练集，需要的仅仅是文档集以及指定主题的数量 k 即可。对于每一个主题均可找出一些词语来描述它。

LDA是一种典型的词袋模型，即它认为一篇文档是由一组词构成的一个集合，词与词之间没有顺序以及先后的关系。一篇文档可以包含多个主题，文档中每一个词都由其中的一个主题生成。

LDA 模型是概率图模型，特点是以狄利克雷分布为多项式分布的先验分布，学习过程就是给定文本集合，通过后验概率分布的估计，推断模型的所有参数。

可以认为LDA是概率潜在语义分析(PLSA)的扩展，在文本生成过程中，LDA使用狄利克雷分布作为先验分布，而PLSA不使用先验分布(或者说假设先验分布是均匀分布)。LDA的优点是：使用先验概率分布，可以防止学习过程中产生的过拟合 。


<br>

**LDA 与 PLSA 异同**
- 相同点
    - 两者都假设话题是单词的多项分布，文本是话题的多项分布。

- 不同点：
    - 在文本生成过程中，LDA使用狄利克雷分布作为先验分布，而PLSA不使用先验分布(或者说假设先验分布是均匀分布。
    - 使用先验概率分布，可以防止学习过程中产生的过拟合 。



### Latent Dirichlet Allocation (LDA) brief introduction

> ref: [What is Latent Dirichlet Allocation (LDA) in NLP?](https://www.analyticssteps.com/blogs/what-latent-dirichlet-allocation-lda-nlp)

Use for Topic modelling

establish semantic relationships between words, find documents or books through a text summary, enhance customer service


> ref: [LDA文本主题模型的学习笔记](https://blog.csdn.net/qq_38556984/article/details/107571714)

LDA是一个无监督分析，对于自然语言处理，可以用来主题分析、关键词提取、文档聚类，计算语义相似度；对于计算机视觉，可以用来图像聚类；换句话说，只要能引入隐变量（主题就是一个隐变量）的场景，都可以尝试。

提出者在一开始建模的时候，有一个假设是：我们在一个单词一个单词敲打出一个文档过程中，每当我们想预先写出一个单词时，其实我们心已有一个定好的主题，比方说我们想写一个主题为体育的新闻，那么我们在体育这个主题下再去找我们相对应的一个单词，那么我们就可能敲出“足球”这个单词；然后我们再去体育这个主题下，敲出来“比赛”这个单词；以此类推，我们生成了整篇文章。这也是LDA模型work的原因。

LDA模型的输出如上图所两个分布，一个是主题分布，另外一个是单词分布。

在每个主题下，所有单词概率加起来为1；
在每个文章下，所有主题的概率加起来为1。

以矩阵的观点来看，输入的是一个N * M 的文章矩阵，其中 N 是文章的数量，M 是单词的数量。
输出的是一个N * K 的主题分布，其中 K 是主题个数，里面每一行代表的是每篇文章的主题分布；
另外一个输出是 M * K的单词分布，里面的每一列对应着每个主题的单词分布

可以从下面几个方向来理解LDA：
1. 概率图模型：数据生成过程定义了观测随机变量和隐藏随机变量的联合概率分布。通过使用联合分布来计算在给定观测变量下隐藏变量的条件分布（后验分布）来进行数据分析。 对于LDA来说，观测变量就是文档中的词；隐藏变量就是主题结构。那么推测文档中隐藏的 主题结构的问题其实就是计算在给定文档下隐藏变量的条件分布（后验分布）。
2. 矩阵分解：单词 * 文档 分解为 单词 * 主题 + 文档 * 主题
3. 聚类：根据主题进行聚类。
4. 降维：每一篇文档可看作关于主题的分布。



There are three hyperparameters in LDA:
- Document-topic density factor (‘α’)
- Topic-word density factor (‘β’)
- The number of topics to be considered (K).


Two fundamental assumptions are made by the LDA:

- Documents are made up of a variety of themes, while topics are made up of a variety of tokens (or words)

- The probability distribution is used to produce the words in these areas. The documents are known as the probability density (or distribution) of subjects, and the topics are known as the probability density (or distribution) of words in statistical terms.


Applications of LDA:

Traditionally, LDA has been used to detect thematic word clusters or subjects in text data. Aside from that, LDA has been employed as a component in more complex applications. 

- Cascaded LDA for taxonomy construction
- Recommendation system based on LDA:
- Gene Expression Classification   


### 数学相关的知识

LDA涉及到的数学知识有：共轭先验分布 、贝叶斯框架、二项分布、Gamma函数、Beta分布、多项分布、Dirichlet分布、吉布斯采样

- 共轭先验分布

后验分布 = 似然分布 * 先验分布  
在贝叶斯概率理论中，如果后验分布和先验分布满足同样的分布律，那么两者称为共轭分布。而先验分布叫作似然函数的共轭先验分布。

$$
p(\theta | x) = \frac{p(x | \theta)  p(\theta)}{p(x)} \propto p(x | \theta)  p(\theta)
$$

- 贝叶斯框架

先验分布 + 数据（似然）$=>$ 后验分布

好人与坏人的例子：   
先验分布：100个好人和100个的坏人，即你认为好人坏人各占一半   
数据：现在你被2个好人帮助了和1个坏人骗了   
后验分布：102个好人和101个的坏人     

现在你的后验分布里面认为好人比坏人多了。这个后验分布接着又变 成你的新的先验分布，当你被1个好人（数据）帮助了和3个坏人（数 据）骗了后，你又更新了你的后验分布为：103个好人和104个的坏人。 依次继续更新下去。   

- 二项分布

二项分布是N重伯努利分布，$X−B(n,p)$。概率密度公式为：

$$
P(K = k) = \tbinom{n}{k} p^k (1-p)^{n-k}
$$

- 多项分布

是二项分布扩展到多维的情况. 多项分布是指单次试验中的随机变量的取值不再是0 − 1 0-10−1的，而是有多种离散值可能（ 1 , 2 , 3... , k ） （1,2,3...,k）（1,2,3...,k）.概率密度函数为：
$$
p([x_1, x_2, x_3, ..., x_k], n, [p_1, p_2, p_3, ..., p_k]) = \frac{n!}{x_1! x_2! x_3! ... x_k!}p_1^{x_1}p_2^{x_2}...p_k^{x_k}
$$

- Gamma函数与Beta分布

Gamma函数可以看成是阶乘在实数集上的延拓，欧拉应用各种参数替换数学技巧与极 限思想，成功推导出Gamma函数：

$$
\Gamma (x) = \int_0^\infty t^{x-1}e^{-t} dt
$$

对于参数 $\alpha > 0, \beta > 0$ , 取值范围为 $[0,1]$ 的随机变量 $x$ 的概率密度函数为:

$$
f(x; \alpha, \beta) = \frac{1}{B(\alpha, \beta)} x^{\alpha-1} (1-x)^{\beta-1}
$$

B的表达式可以由刚才定义的Gamma函数来表达：

$$
\frac{1}{B(\alpha, \beta)} = \frac{\Gamma (\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)}
$$


- Dirichlet分布

Beta分布的多维形式我们一般称他为狄克雷（Dirichlet）分布，所以Dirichlet分布是Beta分布的一种拓展：

$$
f(x_1, x_2, ..., x_k; \alpha_1, \alpha_2 ..., \alpha_k) = \frac{1}{B(\alpha)} \prod \limits_{k} \limits^{i = 1} x_i^{\alpha^i-1}
$$

B的表达式仍可以由刚才定义的Gamma函数来表达：

$$
B(\alpha) = \frac{\prod _{k} ^{i = 1} \Gamma(\alpha_i)}{\Gamma()\sum^k_{i=1}\alpha_i}, \sum \limits_{i=1} \limits^k x^i = 1
$$

至此，所有的reference为
> ref: [如何理解LDA](https://blog.csdn.net/qq_38556984/article/details/107571714)

> ref: [LDA数学八卦](https://www.cnblogs.com/gasongjian/p/7631978.html)

> ref: [浅谈狄利克雷分布——Dirichlet Distribution](https://blog.csdn.net/philthinker/article/details/111999552)

> ref: [文本主题模型之LDA(一) LDA基础](https://www.cnblogs.com/pinard/p/6831308.html)

> ref: [干货 | 一文详解隐含狄利克雷分布（LDA）](http://www.sohu.com/a/239937665_633698)

> Courses: [徐亦达机器学习课程Dirichlet Process](https://youtu.be/qT6CQ9BFDL8)


- 吉布斯采样

概率图模型中最常用的采样技术是马尔可夫链脸蒙特卡罗(Markov chain Monte Carlo, MCMC)

> Courses: [徐亦达机器学习课程 Markov Chain Monte Carlo](https://youtu.be/s8w8AsFK77c)




### dirichlet distribution

dirichlet distribution <-- beta distribution <-- Bernoulli process <-- Bernoulli trial

> ref: https://www.zhihu.com/question/26751755

> ref: https://blog.csdn.net/philthinker/article/details/111999552

Dirichlet分布是Beta分布的多元推广。Beta分布是二项式分布的共轭分布，Dirichlet分布是多项式分布的共轭分布。通常情况下，我们说的分布都是关于某个参数的函数，把对应的参数换成一个函数（函数也可以理解成某分布的概率密度）就变成了关于函数的函数。

> ref: https://www.zhihu.com/question/26751755/answer/147053143



