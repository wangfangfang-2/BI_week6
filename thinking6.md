XGBoost与GBDT的区别是什么？
xgboost是Gradient Boosting的工程版本，是一种高效系统实现，并不是一种单一算法。xgboost里面的基学习器除了用tree(gbtree)，也可用线性分类器(gblinear)。而GBDT则特指梯度提升决策树算法。
xgboost相对于普通gbm的实现，可能具有以下的一些优势：
显式地将树模型的复杂度作为正则项加在优化目标
公式推导里用到了二阶导数信息，而普通的GBDT只用到一阶
允许使用column(feature) sampling来防止过拟合，借鉴了Random Forest的思想，sklearn里的gbm也有类似实现。
4.实现了一种分裂节点寻找的近似算法，用于加速和减小内存消耗。
5.节点分裂算法能自动利用特征的稀疏性。
6.data事先排好序并以block的形式存储，利于并行计算
7.cache-aware, out-of-core computation，这个我不太懂。。
8.支持分布式计算可以运行在MPI，YARN上，得益于底层支持容错的分布式通信框架rabit。