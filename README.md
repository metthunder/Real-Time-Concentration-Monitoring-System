# Real-Time Ammonia Concentration Monitoring System

This project monitors real-time ammonia (NH₃) concentration levels using the MQ-137 sensor connected to an Arduino UNO.  
In addition to live monitoring, an LSTM-based model is trained on historical data to predict future concentration trends.  
The predictions help in taking early action in case of dangerous gas levels.

## 🌟 Project Highlights

- **Sensor Data Acquisition**: Real-time ammonia concentration readings via MQ-137 + Arduino UNO.
- **LSTM-Based Forecasting**: Predicts future ammonia concentrations using a model trained on past data.
- **Web Dashboard**: Displays live sensor readings and predictions through an interactive web dashboard.
- **Alerts**: Monitor rising ammonia levels easily with predictive trends.

## Hardwares Used

- **MQ-137 Ammonia Gas Sensor
- **Arduino UNO
- **USB Cable for Arduino-PC communication

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/metthunder/Real-Time-Concentration-Monitoring-System
cd Real-Time-Concentration-Monitoring-System
```

### 2. Installing Dependencies

```bash
pip install -r requirements.txt
```

### 3. Running the Dashboard

```bash
python "Final Dashboard.py"
```

After running, open your browser and navigate to: https://127.0.0.1:8052

You will see the real-time ammonia monitoring dashboard, showing:
Live sensor readings
Future concentration predictions based on the LSTM model
