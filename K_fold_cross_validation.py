import numpy as np
from sklearn.datasets import load_iris # 依旧鸢尾花数据集
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

X = load_iris().data
y = load_iris().target
print(f"数据集大小: {X.shape[0]} 样本数, {X.shape[1]} 特征数")

K = 5 # 选择5折交叉验证
kf = KFold(n_splits=K, shuffle=True, random_state=42)

"""
这段代码实现了一个完整的 5 折交叉验证过程。
具体来说，它利用 KFold 将数据集 X, y 随机划分成 5 个互斥的子集（折），
然后循环 5 次：每次选取其中一折作为验证集，其余 4 折作为训练集，
用逻辑回归模型在训练集上拟合，并在验证集上预测标签，计算准确率。
每个折的准确率被记录在 scores 列表中，循环结束后我们可以得到 5 个准确率，
它们的平均值和标准差就反映了模型在不同数据子集上的泛化性能。
整个过程避免了因单次数据划分带来的偶然性，让我们对模型的稳定性和泛化能力有更可靠的估计。
"""
scores = []
for fold, (train_index, val_index) in enumerate(kf.split(X)):
    x_train, x_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    model = LogisticRegression(max_iter=500) # 增加迭代次数以确保收敛
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val) # 预测验证集标签
    acc = accuracy_score(y_val, y_pred) # 计算准确率
    scores.append(acc) # 记录每折的准确率
    
    print(f"第 {fold + 1} 折, {len(x_train)}个样本, {len(x_val)}个验证样本, 准确率: {acc:.4f}")

print(f"平均准确率: {np.mean(scores):.4f}, 标准差: {np.std(scores):.4f}")