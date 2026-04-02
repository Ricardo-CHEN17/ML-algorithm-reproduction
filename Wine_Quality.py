import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim

# 从 UCI 官网直接读取红葡萄酒数据集
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')

# 数据预处理
df['quality'] = df['quality'].apply(lambda x : 1 if x >= 6 else 0)

X = df.drop('quality', axis=1).values
y = df['quality'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_t = torch.FloatTensor(X_train)
X_test_t = torch.FloatTensor(X_test)
y_train_t = torch.FloatTensor(y_train).view(-1, 1)
y_test_t = torch.FloatTensor(y_test).view(-1, 1)

class WineNet(nn.Module):
    def __init__(self, input_size):
        super(WineNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x
    
model = WineNet(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
epochs = 100
losses = []

for epoch in range(epochs):
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()    
    
    losses.append(loss.item())
    if (epoch+1) % 40 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

model.eval() # 切换到预测模式
with torch.no_grad():
    y_pred_prob = model(X_test_t)
    # 概率 > 0.5 判定为 1 (好酒)，否则为 0
    y_pred = (y_pred_prob > 0.5).float()
    
    # 计算准确率
    acc = accuracy_score(y_test_t.numpy(), y_pred.numpy())
    print("\n--- 模型评估报告 ---")
    print(f"准确率 (Accuracy): {acc * 100:.2f}%")
    print("\n分类详细报告:")
    print(classification_report(y_test_t.numpy(), y_pred.numpy(), target_names=['Bad Wine', 'Good Wine']))

# 绘制 Loss 下降曲线
plt.plot(range(epochs), losses)
plt.title('Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='quality', data=df)
plt.title('Wine Quality Distribution')
plt.xlabel('Quality (0 = Bad, 1 = Good)')
plt.ylabel('Count')
plt.show()