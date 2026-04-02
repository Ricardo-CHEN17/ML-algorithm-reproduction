import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

# 依旧手搓数据集，包含文本和标签
data = {
    'text': [
        "How to build a web API with Python?",
        "Debugging neural networks in PyTorch",
        "Connecting FastAPI to a MySQL database",
        "What is the difference between linear regression and deep learning?",
        "Deploying a Python machine learning model as a web service"
    ],
    'tags': [
        ['python', 'fastapi'],
        ['python', 'machine-learning', 'pytorch'],
        ['python', 'fastapi', 'mysql'],
        ['machine-learning'],
        ['python', 'machine-learning', 'fastapi']
    ]
}
df = pd.DataFrame(data)

# 文本特征提取
vectorizer = TfidfVectorizer(max_features=50)
X_features = vectorizer.fit_transform(df['text']).toarray()

# 标签二值化
mlb = MultiLabelBinarizer()
y_labels = mlb.fit_transform(df['tags'])

print(f"识别出的所有可能标签: {mlb.classes_}")
print(f"第一个样本的特征维度: {X_features[0].shape}")
print(f"第一个样本的真实标签矩阵: {y_labels[0]}")

class TagDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
# 实例化 Dataset 和 DataLoader
dataset = TagDataset(X_features, y_labels)
# 因为数据少，把 batch_size 设为 2 感受一下批处理
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 搭建神经网络
class MultiLabelClassifier(nn.Module):
    def __init__(self, inputdim, hidden_dim, output_dim):
        super(MultiLabelClassifier, self).__init__()
        self.fc1 = nn.Linear(inputdim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 获取输入维度 (词汇表大小) 和 输出维度 (标签种类数)
INPUT_DIM = X_features.shape[1]
OUTPUT_DIM = len(mlb.classes_)
HIDDEN_DIM = 16

# 初始化模型、损失函数和优化器
model = MultiLabelClassifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
criterion = nn.BCEWithLogitsLoss() # 核心：多标签损失函数
optimizer = optim.Adam(model.parameters(), lr=0.05)

epochs = 50
print("开始训练...")
for epoch in range(epochs):
    total_loss = 0
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")

print("\n--- 训练完成，开始预测新问题 ---")
model.eval() # 切换到评估模式

# 假设用户提出了一个新问题
new_question = ["I want to build a deep learning API using FastAPI and Python"]
new_feature = vectorizer.transform(new_question).toarray()
new_tensor = torch.tensor(new_feature, dtype=torch.float32)

with torch.no_grad():
    raw_output = model(new_tensor)
    # 推理时，我们需要手动加上 Sigmoid，把输出变成 0~1 的概率
    probabilities = torch.sigmoid(raw_output)

# 设定一个阈值（比如 0.5），概率大于 0.5 的我们认为该标签激活
threshold = 0.5
predicted_labels = (probabilities > threshold).int().numpy()[0]

# 将 [0, 1, 0...] 还原回文本标签
predicted_tags = mlb.inverse_transform(predicted_labels.reshape(1, -1))

print(f"新问题: {new_question[0]}")
print(f"各个标签的预测概率: {probabilities.numpy()[0]}")
print(f"最终预测打上的标签: {predicted_tags[0]}")