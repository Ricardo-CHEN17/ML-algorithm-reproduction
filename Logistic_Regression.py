import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42) 
x = np.random.rand(100, 1)
y = 5 * x + np.random.randn(100, 1)

plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Simulated Data")
plt.show()

w = np.random.randn(1)[0]
b = np.random.randn(1)[0]

def forward(w, x, b):
    return w * x + b

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_log_loss(y_pred, y_true):
    m = len(y_true)
    return (-1 / m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def compute_gradients(x, y_true, y_pred):
    m = len(y_true)
    dw = (1 / m) * np.sum((y_pred - y_true) * x)
    db = (1 / m) * np.sum(y_pred - y_true)
    return dw, db

learning_rate = 0.01
epochs = 100
losses = []

print(f"初始参数: w = {w:.4f}, b = {b:.4f}")

for epoch in range(epochs):
    y_pred = sigmoid(forward(w, x, b))
    
    loss = compute_log_loss(y_pred, y)
    losses.append(loss)
    dw, db = compute_gradients(x, y, y_pred)
    
    # 梯度下降：更新参数
    w = w - learning_rate * dw
    b = b - learning_rate * db

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - w: {w:.4f}, b: {b:.4f}")

print(f"训练完成! 最终参数: w = {w:.4f}, b = {b:.4f}")

plt.figure(figsize=(12, 5))

# 第一张图：数据散点图与拟合直线
plt.subplot(1, 2, 1)
plt.scatter(x, y, label="Actual Data")
plt.plot(x, sigmoid(forward(w, x, b)), color='red', label="Fitted Line")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Logistic Regression Fit")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(epochs), losses, color='blue')
plt.xlabel("Epochs")
plt.ylabel("Log Loss")
plt.title("Training Loss")
plt.show()