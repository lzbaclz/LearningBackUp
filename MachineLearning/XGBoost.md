# The study of ML model XGBoost

XGBoost是一个优化的分布式梯度增强库，旨在实现高效，灵活和便携。在 Gradient Boosting 框架下实现机器学习算法。XGBoost提供并行树提升（也称为GBDT，Gradient Boosting Decision Tree）可以快速准确地解决许多数据科学问题。

XGBoost是对梯度提升算法的改进,求解损失函数极值时使用了牛顿法，将损失函数泰勒展开到二阶，另外损失函数中加入了正则化项。训练时的目标函数由两部分构成，第一部分为梯度提升算法损失，第二部分为正则化项。

> read from [wikipedia](https://en.wikipedia.org/wiki/XGBoost)

XGBoost (eXtreme Gradient Boosting) is an open-source software library which provides a regularizing gradient boosting framework for programming language, such as C++, Java, Python, R, Ruby etc.

It aims to provide a "Scalable, Portable and Distributed Gradient Boosting (GBM, GBRT, GBDT) Library".

XGBoost 即极端梯度提升，是可扩展的分布式梯度提升决策树 (GBDT) 机器学习库。XGBoost 提供并行树提升功能，是用于回归、分类和排名问题的先进机器学习库。

