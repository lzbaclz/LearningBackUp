# Some definitions about machine learning in 19-21 July


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