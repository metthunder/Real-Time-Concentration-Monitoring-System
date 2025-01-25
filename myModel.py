import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
from datetime import datetime
import os

class NH3ModelTrainer:
    def __init__(self, window_size=180, prediction_steps=50):
        self.window_size = window_size
        self.prediction_steps = prediction_steps
        self.scaler = MinMaxScaler(feature_range=(0, 1))  # Scaling between 0 and 1
        self.model = None

    def handle_outliers(self, data):
        """Handles outliers by replacing with rolling mean."""
        df = data.copy()
        
        # Use rolling window to handle outliers
        rolling_mean = df['concentration'].rolling(window=60, center=True).mean()
        rolling_std = df['concentration'].rolling(window=60, center=True).std()
        
        # Define outliers as values beyond 3 standard deviations
        mask = (df['concentration'] < rolling_mean - 3 * rolling_std) | (df['concentration'] > rolling_mean + 3 * rolling_std)
        df.loc[mask, 'concentration'] = rolling_mean[mask]
        
        return df

    def preserve_trends(self, data):
        """Smooths trends using exponential smoothing."""
        df = data.copy()
        df['concentration'] = df['concentration'].ewm(span=30, adjust=False).mean()
        return df

    def prepare_sequences(self, data):
        """Prepares data sequences for LSTM."""
        cleaned_data = self.handle_outliers(data)
        processed_data = self.preserve_trends(cleaned_data)
        
        # Scale the concentration data
        scaled_data = self.scaler.fit_transform(processed_data['concentration'].values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(len(scaled_data) - self.window_size - self.prediction_steps):
            X.append(scaled_data[i:(i + self.window_size)])
            y.append(scaled_data[(i + self.window_size):(i + self.window_size + self.prediction_steps)])
            
        return np.array(X), np.array(y)

    def build_model(self):
        """Builds the LSTM model with tuned parameters."""
        model = Sequential([
            LSTM(128, activation='relu', input_shape=(self.window_size, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(64, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(self.prediction_steps)
        ])
        
        model.compile(optimizer='adam', loss='huber', metrics=['mae'])
        return model

    def train(self, data_path, epochs=50, batch_size=32):
        """Trains the model on chronological data."""
        # Load and prepare data
        df = pd.read_csv(data_path)
        
        # Convert timestamp to datetime if it's not already
        if df['timestamp'].dtype == 'object':
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp to ensure chronological order
        df = df.sort_values('timestamp')
        
        # Prepare sequences
        X, y = self.prepare_sequences(df)
        
        # Split chronologically (use the last 20% as validation set)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Build and train model
        self.model = self.build_model()
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        return history

    def save_models(self, path='models'):
        """Saves the trained model and scaler."""
        if not os.path.exists(path):
            os.makedirs(path)
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f"{path}/lstm_model_{timestamp}.h5"
        scaler_path = f"{path}/scaler_{timestamp}.pkl"
        
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        
        return {
            'model_path': model_path,
            'scaler_path': scaler_path
        }

# Example usage
if __name__ == "__main__":
    # Path to your ammonia concentration data CSV
    data_path = "nh3_sample_data.csv"
    
    # Initialize and train the model
    trainer = NH3ModelTrainer(window_size=180, prediction_steps=50)
    history = trainer.train(data_path)
    
    # Save the trained model and scaler
    model_paths = trainer.save_models()
    
    # Print paths to saved files
    print(f"Model saved to: {model_paths['model_path']}")
    print(f"Scaler saved to: {model_paths['scaler_path']}")
