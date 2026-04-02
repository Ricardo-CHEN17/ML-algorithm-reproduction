import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

np.random.seed(42)
num_samples = 1000
time_of_day = np.random.uniform(0, 24, num_samples)
day_of_week = np.random.randint(1, 8, num_samples)
weather_index = np.random.uniform(0, 1, num_samples)

traffic_volume = 50*np.sin((time_of_day-6)*np.pi/12) + 20 * weather_index + np.random.normal(0, 5, num_samples) + 100

df = pd.DataFrame({
    "Time": time_of_day,
    "Day": day_of_week,
    "Weather": weather_index,
    "Volume": traffic_volume
})

features = ['Time', 'Day', 'Weather']
df[features] = (df[features] - df[features].mean()) / df[features].std()

X = torch.tensor(df[features].values, dtype=torch.float32)
y = torch.tensor(df['Volume'].values, dtype=torch.float32).view(-1, 1)

class TrafficPredictionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Linear(3, 16)
        self.output_layer = nn.Linear(16, 1)
        self.relu = nn.ReLU()
       
    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x
    
model = TrafficPredictionModel()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epchos = 2500
losses = []

for epoch in range(epchos):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}/{epchos} - Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "traffic_prediction_model.pth")

plt.figure(figsize=(8, 4))
plt.plot(losses, color='blue', label='Training Loss')
plt.title("Model Training Progression")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (Loss)")
plt.legend()
plt.grid(True)
plt.show()

model.eval()

with torch.no_grad():
    # 此时模型输出的【直接就是】真实的交通流量，因为你的 y 并没有归一化
    sample_preds_real = model(X[:50]).numpy()
    sample_actuals_real = y[:50].numpy()

plt.figure(figsize=(10, 5))
plt.plot(sample_actuals_real, label='Actual Traffic Volume', marker='o')
plt.plot(sample_preds_real, label='Predicted Traffic Volume', marker='x', linestyle='--')
plt.title("Actual vs. Predicted Traffic Volume (First 50 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Traffic Volume")
plt.legend()
plt.show()