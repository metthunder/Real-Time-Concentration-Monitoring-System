import pandas as pd
import numpy as np
import random
import os

# Constants for the dataset
seconds_in_day = 86400  # 24 hours * 60 minutes * 60 seconds
threshold = 400  # Threshold value for NH3
min_value = 40  # Minimum realistic value for NH3
random.seed(42)
np.random.seed(42)

# Helper function to generate trends dynamically
def generate_trend_dynamic(base_value, trend_type, length, min_value, noise_level=0.05):
    if trend_type == "increasing":
        trend = np.linspace(base_value, base_value + length * noise_level, length)
    elif trend_type == "decreasing":
        trend = np.linspace(base_value, max(base_value - length * noise_level, min_value), length)
    elif trend_type == "steady":
        trend = np.full(length, base_value)
    elif trend_type == "spike":
        trend = np.full(length, base_value)
        spike_index = random.randint(0, length - 1)
        trend[spike_index] += random.uniform(100, 200)  # Large spike
    elif trend_type == "wave":
        # Simulate oscillations
        trend = base_value + 50 * np.sin(np.linspace(0, 2 * np.pi, length))
    else:
        raise ValueError("Invalid trend type")
    # Add random noise
    noise = np.random.normal(0, base_value * noise_level, length)
    return np.clip(trend + noise, min_value, None)  # Ensure values stay above min_value

# Initialize the dataset
data = {
    "Timestamp": pd.date_range(start="2025-01-22", periods=seconds_in_day, freq="s"),
    "NH3": []
}

# Generate dynamic trends for NH3
base_value = threshold * 0.5  # Start at 50% of the threshold value
readings = []

i = 0
while i < seconds_in_day:
    # Decide trend type and length
    trend_type = random.choice(["increasing", "decreasing", "steady", "spike", "wave"])
    length = random.randint(5, 600)  # Mix of short and long trends

    # Avoid exceeding the day's total seconds
    if i + length > seconds_in_day:
        length = seconds_in_day - i

    # Generate the trend
    trend = generate_trend_dynamic(base_value, trend_type, length, min_value)

    # Add the trend to readings
    readings.extend(trend)

    # Update the base value and index
    base_value = max(trend[-1], min_value)  # Ensure base value stays above min_value
    i += length

# Add outliers for NH3
outlier_count = 200  # Increased outliers
outlier_indices = random.sample(range(seconds_in_day), k=outlier_count)
for idx in outlier_indices:
    readings[idx] = random.uniform(400, 600)  # NH3 outliers above the threshold

# Append readings to the dataset
data["NH3"] = readings

# Create the DataFrame
df = pd.DataFrame(data)

# Save the dataset to a valid directory
output_dir = "c:/Users/yatha/Documents/Hilti/"
os.makedirs(output_dir, exist_ok=True)

file_path = os.path.join(output_dir, "nh3_dynamic_trends_dataset.csv")
df.to_csv(file_path, index=False)

print(f"Dynamic NH3 dataset with varied trends saved at: {file_path}")
