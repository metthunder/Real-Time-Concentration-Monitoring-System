import pandas as pd
import numpy as np
import random
import os

# Constants for the dataset
seconds_in_day = 86400  # 24 hours * 60 minutes * 60 seconds
thresholds = {"Cobalt": 0.05, "Nickel": 20, "NH3": 400}  # Threshold values
minimum_values = {"Cobalt": 0.01, "Nickel": 5, "NH3": 50}  # Minimum significant values
random.seed(42)
np.random.seed(42)

# Helper function to generate realistic trends
def generate_trend_safe(base_value, trend_type, length, min_value, noise_level=0.01):
    if trend_type == "increasing":
        trend = np.linspace(base_value, base_value + length * noise_level, length)
    elif trend_type == "decreasing":
        trend = np.linspace(base_value, max(base_value - length * noise_level, min_value), length)
    elif trend_type == "steady":
        trend = np.full(length, base_value)
    elif trend_type == "spike":
        trend = np.full(length, base_value)
        trend[length // 2] += base_value * 2  # Add a spike in the middle
    else:
        raise ValueError("Invalid trend type")
    # Add random noise, ensuring scale is non-negative
    noise_scale = max(base_value * 0.05, 0.001)  # Minimum noise scale
    noise = np.random.normal(0, noise_scale, length)
    return np.clip(trend + noise, min_value, None)  # Ensure values stay above min_value

# Initialize the dataset
data = {
    "Timestamp": pd.date_range(start="2025-01-22", periods=seconds_in_day, freq="s"),
    "Cobalt": [],
    "Nickel": [],
    "NH3": []
}

# Generate trends for each component
for component, threshold in thresholds.items():
    base_value = threshold * 0.5  # Start at 50% of the threshold value
    min_value = minimum_values[component]  # Minimum significant value for the component
    readings = []

    i = 0
    while i < seconds_in_day:
        # Decide trend type and length
        trend_type = random.choice(["increasing", "decreasing", "steady", "spike"])
        length = random.randint(5, 300)  # Trend length between 5 seconds to 5 minutes

        # Avoid exceeding the day's total seconds
        if i + length > seconds_in_day:
            length = seconds_in_day - i

        # Generate the trend
        trend = generate_trend_safe(base_value, trend_type, length, min_value)

        # Add the trend to readings
        readings.extend(trend)

        # Update the base value and index
        base_value = max(trend[-1], min_value)  # Ensure base value stays above min_value
        i += length

    # Add outliers for this component
    outlier_count = 100  # Number of outliers for each component
    outlier_indices = random.sample(range(seconds_in_day), k=outlier_count)
    for idx in outlier_indices:
        readings[idx] = random.uniform(threshold * 2, threshold * 10)  # Extreme outlier values

    # Append readings to the dataset
    data[component] = readings

# Create the DataFrame
df = pd.DataFrame(data)

# Save the dataset to a valid directory
output_dir = "c:/Users/yatha/Documents/Hilti/"
os.makedirs(output_dir, exist_ok=True)

file_path = os.path.join(output_dir, "real_time_metal_monitoring_dataset_full_trends.csv")
df.to_csv(file_path, index=False)

print(f"Dataset with trends for all components saved at: {file_path}")
