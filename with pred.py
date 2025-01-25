import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import time
import serial
import numpy as np
import atexit
from tensorflow.keras.models import load_model
import joblib

class DataStore:
    def __init__(self):
        self.x_data = [time.time()]
        self.y_data = [0]
        self.ser = None
        self.threshold = 25  # ppm
        self.exposure_start_time = None
        self.total_exposure_time = 0
        self.arduino_connected = False
        self.currently_exposed = False

        # Load model and scaler
        try:
            self.model = load_model('models/lstm_model.h5')
            self.scaler = joblib.load('models/scaler.pkl')
            self.window_size = 60  # 1 minutes of data
            self.prediction_steps = 50
            self.model_loaded = True
            print("Model and scaler loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False

    def initialize_serial(self):
        if self.ser is None or not self.ser.is_open:
            try:
                self.ser = serial.Serial("COM7", 9600, timeout=1)
                time.sleep(2)
                self.arduino_connected = True
                print("Successfully connected to Arduino")
                return True
            except serial.SerialException as e:
                print(f"Error connecting to Arduino: {e}")
                self.arduino_connected = False
                return False
        return True

    def read_data(self):
        if not self.arduino_connected:
            self.initialize_serial()
            
        if self.ser and self.ser.is_open:
            try:
                if self.ser.in_waiting > 0:
                    data = self.ser.readline().decode('utf-8').strip()
                    try:
                        return float(data)  # Data already in PPM
                    except ValueError:
                        print(f"Invalid data received: {data}")
            except serial.SerialException as e:
                print(f"Error reading from Arduino: {e}")
                self.arduino_connected = False
                if self.ser and self.ser.is_open:
                    self.ser.close()
                self.ser = None
        return None

    def make_prediction(self):
        """Make predictions using the loaded model"""
        if not self.model_loaded or len(self.y_data) < self.window_size:
            return None
            
        try:
            # Get the last window_size measurements
            recent_data = np.array(self.y_data[-self.window_size:])
            
            # Reshape data for scaling
            recent_data = recent_data.reshape(-1, 1)
            
            # Scale the data
            scaled_data = self.scaler.transform(recent_data)
            
            # Reshape for LSTM input (samples, time steps, features)
            model_input = scaled_data.reshape(1, self.window_size, 1)
            
            # Make prediction
            scaled_prediction = self.model.predict(model_input, verbose=0)
            
            # Reshape prediction for inverse transform
            scaled_prediction = scaled_prediction.reshape(-1, 1)
            
            # Inverse transform
            predictions = self.scaler.inverse_transform(scaled_prediction)
            
            # Ensure predictions are positive and reasonable
            predictions = np.maximum(predictions, 0)
            
            return predictions.flatten()
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

    def cleanup(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial connection closed.")

# Initialize Dash app
app = dash.Dash(__name__)

# Create global data store
data_store = DataStore()

# Register cleanup function
atexit.register(data_store.cleanup)

# Styles
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

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Real-Time NH₃ Concentration Monitor", 
                style={'textAlign': 'center', 'color': '#4CAF50', 'padding': '20px'}),
        html.Div(id='connection-status', 
                 style={'textAlign': 'center', 'margin': '10px'})
    ], style={'backgroundColor': 'white', 'marginBottom': '20px'}),

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
    current_concentration = data_store.read_data()
    current_time = time.time()
    
    if data_store.arduino_connected:
        connection_status = html.Div("Arduino Connected", 
                                   style={'color': 'green', 'fontWeight': 'bold'})
    else:
        connection_status = html.Div("Arduino Not Connected", 
                                   style={'color': 'red', 'fontWeight': 'bold'})

    if current_concentration is None:
        if len(data_store.y_data) > 0:
            current_concentration = data_store.y_data[-1]
        else:
            current_concentration = 0

    # Update exposure time calculation
    if current_concentration > data_store.threshold:
        if not data_store.currently_exposed:
            data_store.exposure_start_time = current_time
            data_store.currently_exposed = True
        elif data_store.exposure_start_time is not None:
            data_store.total_exposure_time = (data_store.total_exposure_time + 
                                            (current_time - data_store.exposure_start_time))
            data_store.exposure_start_time = current_time
    else:
        data_store.currently_exposed = False
        data_store.exposure_start_time = None

    # Update data arrays
    data_store.x_data.append(current_time)
    data_store.y_data.append(current_concentration)

    if len(data_store.x_data) > 180:  # Keep 3 minutes of data
        data_store.x_data.pop(0)
        data_store.y_data.pop(0)

    # Make prediction
    predictions = data_store.make_prediction()

    # Calculate statistics
    max_concentration = max(data_store.y_data)
    min_concentration = min(data_store.y_data)
    avg_concentration = np.mean(data_store.y_data)
    std_dev = np.std(data_store.y_data)

    # Create time arrays for plotting
    time_array = [time.strftime('%H:%M:%S', time.localtime(x)) for x in data_store.x_data]
    
    # Create the figure
    figure = {
        'data': [
            # Actual data trace
            go.Scatter(
                x=time_array,
                y=data_store.y_data,
                mode='lines+markers',
                name='Actual',
                line=dict(color='#4CAF50', width=2),
                marker=dict(size=6)
            ),
            # Threshold line
            go.Scatter(
                x=[time_array[0], time_array[-1]],
                y=[data_store.threshold, data_store.threshold],
                mode='lines',
                name='Threshold',
                line=dict(color='#ff9800', width=2, dash='dash')
            )
        ],
        'layout': go.Layout(
            title='Real-time NH₃ Concentration with Predictions',
            xaxis=dict(
                title='Time',
                gridcolor='#eee',
                showgrid=True
            ),
            yaxis=dict(
                title='Concentration (ppm)',
                gridcolor='#eee',
                range=[0, max(max(data_store.y_data) * 1.2, data_store.threshold * 1.2)],
                showgrid=True
            ),
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='x unified'
        )
    }

    # Add predictions if available
    if predictions is not None and len(predictions) > 0:
        future_times = [current_time + i for i in range(1, len(predictions) + 1)]
        future_times_str = [time.strftime('%H:%M:%S', time.localtime(t)) for t in future_times]
        
        # Add prediction trace
        figure['data'].append(
            go.Scatter(
                x=future_times_str,
                y=predictions,
                mode='lines',
                name='Predicted',
                line=dict(
                    color='rgba(76, 175, 80, 0.5)',
                    width=2,
                    dash='dot'
                ),
                fill='tonexty',
                fillcolor='rgba(76, 175, 80, 0.1)'
            )
        )

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