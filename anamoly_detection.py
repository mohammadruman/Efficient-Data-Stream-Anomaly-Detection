import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
       
# Generate data
def generatedata():
    try:
        # Generate time points from 0 to 10,000 with a step size of 0.1
        data = np.arange(0, 10000, 0.1)  

        # Create a seasonal pattern using a sine wave to simulate cyclic behavior
        seasonal_component = 20 * np.sin(0.1 * data)  

        # Add random noise to introduce some natural variability
        noise = np.random.normal(0, 2, len(data))  

        # Combine the seasonal pattern, noise, and a baseline value of 50 
        # to simulate a realistic data stream
        data_stream = seasonal_component + noise + 50  

        # Randomly select 5% of the data points to introduce anomalies (sudden spikes)
        anomaly_indices = np.random.choice(len(data), size=int(0.05 * len(data)), replace=False)
        data_stream[anomaly_indices] += np.random.normal(50, 10, len(anomaly_indices))

        return data_stream  

    except Exception as e:
        print(f"Error generating data: {e}")
        return np.array([])  # Return an empty array if generation fails

# Defining a class for detecting anomalies using the Z-Score method.
class AnomalyDetector:
    def __init__(self, threshold=3.0):
        
        """
       The Z-Score algorithm is a statistical method used to detect anomalies by measuring 
       how far a data point deviates from the mean in terms of standard deviations.
        Any data with absolute Z-Score that exceeds the provided threshold 
        will be considered an anomaly. A 3.0 threshold is the default value. 
        As would be expected, increasing the threshold gives fewer reports of anomalies, 
        while decreasing the threshold makes the detector more aggressive.
        Effectiveness
        The Z-Score method is effective for normally distributed data because it 
        identifies outliers based on standard deviations from the mean Threshold Selection
        """
        if threshold <= 0:
            raise ValueError("Threshold must be a positive number.")
        self.threshold = threshold

    def detect(self, data):
        # Detect anomalies based on the Z-Score method.
        if not isinstance(data, (np.ndarray, list)):
            raise TypeError("Data must be a numpy array or a list.")
        if len(data) == 0:
            raise ValueError("Data cannot be empty.")

        # Compute Z-Scores for the entire dataset.
        z_scores = zscore(data)  

        # Find indices where Z-Score exceeds the threshold.
        anomalies = np.where(np.abs(z_scores) > self.threshold)[0] 
        return anomalies 

def real_time_anomaly_detection(data_stream, batch_size=50, update_interval=0.1, window_size=200):
    """
    Simulates real-time data processing with sliding window anomaly detection.
    Displays the data stream with detected anomalies highlighted in real-time.
    """
    if batch_size <= 0 or window_size <= 0:
        raise ValueError("Batch size and window size must be positive.")
    if update_interval <= 0:
        raise ValueError("Update interval must be a positive number.")
    if len(data_stream) == 0:
        raise ValueError("Data stream is empty.")

    # Initialize the anomaly detector with a threshold of 3.0 for Z-Score-based detection.
    anomaly_detector = AnomalyDetector(threshold=3.0)  
    plt.ion()  
    fig, ax = plt.subplots() 

    x, y = [], [] 
    stream_data = []  

    # Process the data stream in small batches to mimic real-time data updates.
    for i in range(0, len(data_stream), batch_size):
        # Extract the current batch of data points.
        batch = data_stream[i:i + batch_size]  
        if len(batch) == 0:
            print(f"Warning: Empty batch encountered at index {i}. Skipping.")
            continue

        # Add the batch to the cumulative stream.
        stream_data.extend(batch)  

        # Use only the latest 'window_size' points to detect recent anomalies.
        window_data = stream_data[-window_size:]
        anomalies = anomaly_detector.detect(window_data)  

        # Updating x and y values for plot
        x.extend(range(i, i + batch_size))
        y.extend(batch)
        ax.clear()  # Clear 

        # Plot the data stream in blue
        ax.plot(x, y, label='Data Stream', color='blue')

        # Highlight detected anomalies in red
        anomalies_x = [x[len(x) - len(window_data) + idx] for idx in anomalies]
        anomalies_y = [y[len(y) - len(window_data) + idx] for idx in anomalies]
        ax.scatter(anomalies_x, anomalies_y, color='red', label='Anomalies')

        ax.legend()  
        plt.pause(update_interval)  

    plt.ioff()  
    plt.show() 

# Main block 
if __name__ == "__main__":
    try:
        # Generate the data stream
        data_stream = generatedata()
        if len(data_stream) == 0:
            raise RuntimeError("Failed to generate data stream. Exiting.")

        # Start real-time anomaly detection on the data stream
        real_time_anomaly_detection(data_stream)

        # Write dependencies to requirements.txt
        with open("requirements.txt", "w") as f:
            f.write("numpy\n")
            f.write("pandas\n")
            f.write("matplotlib\n")
            f.write("scipy\n")

    except Exception as e:
        print(f"An error occurred: {e}")
