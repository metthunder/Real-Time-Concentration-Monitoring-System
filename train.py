import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import math
import time
import torch.nn as nn
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data=pd.read_csv("nh3_trends_dataset.csv")
x=data['Timestamp']
y=data['NH3']

nh3data=np.array(data['NH3'])
q1=np.percentile(nh3data,25)
q3=np.percentile(nh3data,75)
lower=q1-1.5*q3
upper=q3+1.5*q3


window=50

for i in range(len(nh3data)):
    if nh3data[i]<lower or nh3data[i]>upper:
        if i-window>=0 and i+window<len(nh3data):
            window_mean=np.mean(nh3data[i-window:i+window])
            nh3data[i]=window_mean
        elif i-window<0:
            window_mean=np.mean(nh3data[0:i+window])
            nh3data[i]=window_mean
        elif i+window>=len(nh3data):
            window_mean=np.mean(nh3data[i-window:])
            nh3data[i]=window_mean


normalise = MinMaxScaler(feature_range=(-1, 1))
nh3data = normalise.fit_transform(nh3data.reshape(-1,1))


lookback=15
forecast_horizon=10


def split_data(data_raw, lookback, forecast_horizon):
    data = []

    for index in range(len(data_raw) - lookback - forecast_horizon + 1):
        data.append(data_raw[index: index + lookback + forecast_horizon])

    data = np.array(data)
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - test_set_size

    x_train = data[:train_set_size, :lookback, :]
    y_train = data[:train_set_size, lookback:, :]

    x_test = data[train_set_size:, :lookback, :]
    y_test = data[train_set_size:, lookback:, :]

    return [x_train, y_train, x_test, y_test]

x_train, y_train, x_test, y_test = split_data(nh3data, lookback, forecast_horizon)

x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 50
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cpu'

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, forecast_horizon, device):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.device = device

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, forecast_horizon * output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=self.device).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=self.device).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = out[:, -1, :]  
        out = self.fc(out)  
        out = out.view(-1, self.forecast_horizon, self.output_dim)
        return out



model = LSTM(input_dim, hidden_dim, num_layers, output_dim, forecast_horizon, device).to(device)
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

model=model.to(device)

from tqdm import tqdm

hist = np.zeros(num_epochs)
loss_val = []
start_time = time.time()

for epoch in tqdm(range(num_epochs)):
    model.train()
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
    y_train_pred = model(x_train_tensor)
    loss = criterion(y_train_pred, y_train_tensor)
    hist[epoch] = loss.item() 
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    if (epoch + 1) % 10 == 0: 
        print(f"Epoch {epoch + 1}/{num_epochs}, MSE: {loss.item():.4f}")
    loss_val.append(loss.item())

training_time = time.time() - start_time
print(f"Training completed in: {training_time:.2f} seconds")

torch.save(model.state_dict(), 'lstm.pth')
print('Model saved at lstm.pth')

import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(normalise, f)

print('Scaler saved at scaler.pkl')
