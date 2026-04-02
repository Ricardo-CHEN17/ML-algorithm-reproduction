import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42) 
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)


plt.scatter(X, y)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Simulated Data")
plt.show()

w = np.random.randn(1)[0]
b = np.random.randn(1)[0]

def forward(w, x, b):
    return w * x + b

def compute_loss(y_pred, y_true):
    m = len(y_true)
    return (1 / (2 * m)) * np.sum((y_pred - y_true) ** 2)

def compute_gradients(x, y_true, y_pred):
    m = len(y_true)
    dw = (1 / m) * np.sum((y_pred - y_true) * x)
    db = (1 / m) * np.sum(y_pred - y_true)
    return dw, db

learning_rate = 0.1 
epochs = 100
losses = []

print(f"初始参数: w = {w:.4f}, b = {b:.4f}")

for epoch in range(epochs):
    y_pred = forward(X, w, b)
    
    loss = compute_loss(y_pred, y)
    losses.append(loss)
    dw, db = compute_gradients(X, y, y_pred)
    
    # 梯度下降：更新参数
    w = w - learning_rate * dw
    b = b - learning_rate * db

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - w: {w:.4f}, b: {b:.4f}")

print(f"训练完成! 最终参数: w = {w:.4f}, b = {b:.4f}")

plt.figure(figsize=(12, 5))

# 第一张图：数据散点图与拟合直线
plt.subplot(1, 2, 1)
plt.scatter(X, y, label="Actual Data")
plt.plot(X, forward(X, w, b), color='red', label="Fitted Line")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression Fit")
plt.legend()

# 第二张图：损失函数下降曲线
plt.subplot(1, 2, 2)
plt.plot(range(epochs), losses, color='blue')
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Loss curve during training")

plt.tight_layout()
plt.show()