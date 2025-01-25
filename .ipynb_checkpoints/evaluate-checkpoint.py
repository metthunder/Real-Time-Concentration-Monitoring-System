import torch
import torch.nn as nn
import pickle
import numpy as np

MODEL_PATH='lstm.pth'
SCALER_PATH='scaler.pkl'


input_seq = np.array([15,16,17,18,19,20,19,18,17,16,15, 16,17,18,19]).reshape(-1,1)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
input_data = scaler.transform(input_seq)

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

input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
forecast_horizon = 10
device='cpu'

model = LSTM(input_dim, hidden_dim, num_layers, output_dim, forecast_horizon, device).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

x=torch.tensor(input_data, dtype=torch.float32)
x=x.reshape(1,-1,1)
y_pred=model(x).detach().numpy()
y_pred=y_pred.reshape(-1,1)
final_y_pred=scaler.inverse_transform(y_pred)
final_y_pred=final_y_pred.reshape(-1)
print(final_y_pred)