import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import time
import serial
import numpy as np
import atexit
import torch
import torch.nn as nn
import pickle

# LSTM Model Definition
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

# Global variables for data storage
class DataStore:
    def __init__(self):
        self.x_data = [time.time()]
        self.y_data = [0]
        self.ser = None
        self.threshold = 25  # ppm
        self.exposure_start_time = None
        self.total_exposure_time = 0  # in seconds
        self.arduino_connected = False
        self.currently_exposed = False
        
        # LSTM Model Initialization
        self.input_dim = 1
        self.hidden_dim = 32
        self.num_layers = 2
        self.output_dim = 1
        self.forecast_horizon = 10
        self.device = 'cpu'
        
        # Load Scaler and Model
        try:
            with open('scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.model = LSTM(
                self.input_dim, 
                self.hidden_dim, 
                self.num_layers, 
                self.output_dim, 
                self.forecast_horizon, 
                self.device
            ).to(self.device)
            
            self.model.load_state_dict(torch.load('lstm.pth', weights_only=True))
            self.model.eval()
            print("LSTM Model and Scaler loaded successfully")
        except Exception as e:
            print(f"Error loading LSTM model: {e}")
            self.model = None
            self.scaler = None

    def predict_next_datapoints(self, input_seq):
        if self.model is None or self.scaler is None:
            return None
        
        try:
            # Prepare input data
            input_data = self.scaler.transform(input_seq.reshape(-1, 1))
            x = torch.tensor(input_data, dtype=torch.float32).reshape(1, -1, 1)
            
            # Predict
            y_pred = self.model(x).detach().numpy()
            y_pred = y_pred.reshape(-1, 1)
            final_y_pred = self.scaler.inverse_transform(y_pred)
            
            return final_y_pred.reshape(-1)
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

    def initialize_serial(self):
        try:
            # Try multiple potential COM ports
            port = 'COM7'
            self.ser = serial.Serial(port, 9600, timeout=1)
            time.sleep(2)  # Allow Arduino to initialize
            self.arduino_connected = True
            print(f"Successfully connected to Arduino on {port}")
            return True
        
        except Exception as e:
            print(f"Unexpected error in serial connection: {e}")
            self.arduino_connected = False
            return False

    def read_data(self):
        """Read data from Arduino"""
        if not self.arduino_connected:
            self.initialize_serial()
            
        if self.ser and self.ser.is_open:
            try:
                if self.ser.in_waiting > 0:
                    data = self.ser.readline().decode('utf-8').strip()
                    try:
                        return float(data)
                    except ValueError:
                        print(f"Invalid data received: {data}")
            except serial.SerialException as e:
                print(f"Error reading from Arduino: {e}")
                self.arduino_connected = False
                if self.ser and self.ser.is_open:
                    self.ser.close()
                self.ser = None
        return None

    def cleanup(self):
        """Cleanup function"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial connection closed.")

# Create global data store
data_store = DataStore()

# Register cleanup function
atexit.register(data_store.cleanup)

# Initialize Dash app
app = dash.Dash(__name__)

# Define styles
CARD_STYLE = {
    'backgroundColor': 'white',
    'borderRadius': '10px',
    'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
    'padding': '20px',
    'margin': '10px',
    'flex': '1',
    'minWidth': '300px',
    'textAlign': 'center'
}

VALUE_STYLE = {
    'fontSize': '24px',
    'fontWeight': 'bold',
    'margin': '10px 0'
}

LABEL_STYLE = {
    'fontSize': '14px',
    'color': '#666',
    'textTransform': 'uppercase'
}

# Data store to manage state
class DataStore:
    def __init__(self):
        self.threshold = 25
        self.x_data = []
        self.y_data = []
        self.arduino_connected = False
        self.currently_exposed = False
        self.total_exposure_time = 0
        self.exposure_start_time = None
        self.threshold_crossed = False

    def initialize_serial(self):
        # Simulate connection to Arduino (for testing purposes)
        self.arduino_connected = True

    def read_data(self):
        # Simulate reading data from Arduino (for testing purposes)
        return np.random.uniform(20, 30)  # Return random concentration values between 20 and 30

    def predict_next_datapoints(self, input_seq):
        # Simple prediction model (for demonstration)
        return input_seq[-1] + np.random.uniform(-0.5, 0.5, 5)  # Predict next 5 points with random fluctuation

data_store = DataStore()

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Real-Time Ammonia Concentration Monitor", 
                style={'textAlign': 'center', 'color': '#4CAF50', 'padding': '20px'}),
        html.Div(id='connection-status', 
                 style={'textAlign': 'center', 'margin': '10px'})
    ], style={'backgroundColor': 'white', 'marginBottom': '20px'}),

    # Threshold Control Card
    html.Div([
        html.Div([
            html.Div("Threshold Control (ppm)", style=LABEL_STYLE),
            dcc.Input(
                id='threshold-input',
                type='number',
                value=25,
                min=0,
                step=1,
                style={
                    'width': '150px',
                    'height': '40px',
                    'margin': '10px',
                    'padding': '5px 10px',
                    'fontSize': '18px',
                    'borderRadius': '5px',
                    'border': '2px solid #4CAF50'
                }
            ),
            html.Button(
                'Update Threshold', 
                id='threshold-button', 
                style={
                    'backgroundColor': '#4CAF50',
                    'color': 'white',
                    'border': 'none',
                    'padding': '12px 24px',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'fontSize': '16px',
                    'fontWeight': 'bold',
                    'transition': 'background-color 0.3s',
                    'margin': '10px'
                }
            )
        ], style={**CARD_STYLE, 'maxWidth': '400px', 'margin': '20px auto'})
    ], style={'display': 'flex', 'justifyContent': 'center'}),

    # Current Value Display
    html.Div([
        html.Div([
            html.Div("Current Concentration", style=LABEL_STYLE),
            html.Div(id='current-value', style={**VALUE_STYLE, 'color': '#4CAF50'}),
        ], style=CARD_STYLE),
        
        html.Div([
            html.Div("Time Above Threshold", style=LABEL_STYLE),
            html.Div(id='exposure-time', style={**VALUE_STYLE, 'color': '#ff9800'}),
        ], style=CARD_STYLE),
        
        html.Div([
            html.Div("Status", style=LABEL_STYLE),
            html.Div(id='status-indicator', style={**VALUE_STYLE, 'color': '#2196F3'}),
        ], style=CARD_STYLE),
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap', 'margin': '20px 0'}),

    # Graph
    html.Div([
        dcc.Graph(id='live-update-graph', animate=False)
    ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)'}),

    # Statistics Cards
    html.Div(id='stats-div', style={
        'display': 'flex',
        'flexWrap': 'wrap',
        'justifyContent': 'space-around',
        'margin': '20px 0'
    }),

    dcc.Interval(
        id='interval-component',
        interval=1000,  # Update every second
        n_intervals=0
    )
], style={'padding': '20px', 'backgroundColor': '#f5f5f5', 'minHeight': '100vh'})

# Add callback for threshold update
@app.callback(
    Output('threshold-input', 'value'),
    [Input('threshold-button', 'n_clicks')],
    [State('threshold-input', 'value')]
)
def update_threshold(n_clicks, new_threshold):
    if n_clicks is not None:
        data_store.threshold = new_threshold
    return new_threshold

@app.callback(
    [Output('live-update-graph', 'figure'),
     Output('stats-div', 'children'),
     Output('connection-status', 'children'),
     Output('current-value', 'children'),
     Output('exposure-time', 'children'),
     Output('status-indicator', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    # Read data from Arduino
    current_concentration = data_store.read_data()
    current_time = time.time()
    
    # Update connection status
    if data_store.arduino_connected:
        connection_status = html.Div("Arduino Connected", 
                                   style={'color': 'green', 'fontWeight': 'bold'})
    else:
        connection_status = html.Div("Arduino Not Connected", 
                                   style={'color': 'red', 'fontWeight': 'bold'})

    # If no data received, keep the last value
    if current_concentration is None:
        if len(data_store.y_data) > 0:
            current_concentration = data_store.y_data[-1]
        else:
            current_concentration = 0

    # Update exposure time calculation and threshold crossing tracking
    if current_concentration > data_store.threshold:
        if not data_store.currently_exposed:
            data_store.exposure_start_time = current_time
            data_store.currently_exposed = True
            data_store.threshold_crossed = True
        elif data_store.exposure_start_time is not None:
            data_store.total_exposure_time = (data_store.total_exposure_time + 
                                            (current_time - data_store.exposure_start_time))
            data_store.exposure_start_time = current_time
    else:
        data_store.currently_exposed = False
        data_store.exposure_start_time = None
        data_store.threshold_crossed = False

    # Append new data
    data_store.x_data.append(current_time)
    data_store.y_data.append(current_concentration)

    # Maintain only the last 50 data points
    if len(data_store.x_data) > 50:
        data_store.x_data.pop(0)
        data_store.y_data.pop(0)

    # Prediction for next datapoints
    prediction = None
    if len(data_store.y_data) >= 15:  # Ensure we have enough historical data
        input_seq = np.array(data_store.y_data[-15:])
        prediction = data_store.predict_next_datapoints(input_seq)

    # Calculate statistics
    max_concentration = max(data_store.y_data)
    min_concentration = min(data_store.y_data)
    avg_concentration = np.mean(data_store.y_data)
    std_dev = np.std(data_store.y_data)

    # Create the figure
    y_padding = 5
    figure = {
        'data': [
            go.Scatter(
                x=[time.strftime('%H:%M:%S', time.localtime(ts)) for ts in data_store.x_data],
                y=data_store.y_data,
                mode='lines+markers',
                name='Actual Concentration',
                line=dict(color='#4CAF50', width=2),
                marker=dict(size=6)
            ),
            go.Scatter(
                x=[time.strftime('%H:%M:%S', time.localtime(data_store.x_data[0])), 
                   time.strftime('%H:%M:%S', time.localtime(data_store.x_data[-1]))],
                y=[data_store.threshold, data_store.threshold],
                mode='lines',
                name='Threshold',
                line=dict(color='#ff9800', width=2, dash='dash')
            )
        ],
        'layout': go.Layout(
            title='Real-time Concentration Trends',
            xaxis=dict(title='Time', gridcolor='#eee'),
            yaxis=dict(title='Concentration (ppm)', 
                      range=[min(data_store.y_data) - y_padding, max(data_store.y_data) + y_padding],
                      gridcolor='#eee'),
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            transition=dict(duration=500),
        )
    }

    # Add prediction trace if available
    if prediction is not None:
        # Generate x-axis for prediction
        last_time = data_store.x_data[-1]
        pred_times = [last_time + i for i in range(1, len(prediction) + 1)]
        
        prediction_trace = go.Scatter(
            x=[time.strftime('%H:%M:%S', time.localtime(ts)) for ts in pred_times],
            y=prediction,
            mode='lines',
            name='Predicted Concentration',
            line=dict(color='#2196F3', width=2, dash='dot')
        )
        figure['data'].append(prediction_trace)

    # Create statistics cards

    stats = [
        html.Div([
            html.Div("Maximum", style=LABEL_STYLE),
            html.Div(f"{max_concentration:.1f} ppm", style=VALUE_STYLE)
        ], style=CARD_STYLE),
        
        html.Div([
            html.Div("Minimum", style=LABEL_STYLE),
            html.Div(f"{min_concentration:.1f} ppm", style=VALUE_STYLE)
        ], style=CARD_STYLE),
        
        html.Div([
            html.Div("Average", style=LABEL_STYLE),
            html.Div(f"{avg_concentration:.1f} ppm", style=VALUE_STYLE)
        ], style=CARD_STYLE),
        
        html.Div([
            html.Div("Standard Deviation", style=LABEL_STYLE),
            html.Div(f"{std_dev:.2f}", style=VALUE_STYLE)
        ], style=CARD_STYLE),
    ]

    # Total exposure time in minutes and seconds
    total_exposure_minutes = int(data_store.total_exposure_time // 60)
    total_exposure_seconds = int(data_store.total_exposure_time % 60)

    # Current value display
    current_value = f"{current_concentration:.1f} ppm"
    exposure_time = f"{total_exposure_minutes}m {total_exposure_seconds}s"
    status = "NORMAL" if current_concentration <= data_store.threshold else "ALERT"

    return figure, stats, connection_status, current_value, exposure_time, status

if __name__ == '__main__':
    # Initialize serial connection
    data_store.initialize_serial()
    
    # Run the app
    app.run_server(debug=False, port=8051)