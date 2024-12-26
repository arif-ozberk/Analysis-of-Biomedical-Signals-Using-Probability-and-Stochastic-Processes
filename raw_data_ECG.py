import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load ECG data from a CSV file
file_path = './ecg.csv'
data = pd.read_csv(file_path)
first_data = data.head(200)

# Flatten the ECG data into a single array
ecg_signal = first_data.to_numpy().flatten()

# Create a time vector assuming a sampling frequency (e.g., 500 Hz)
fs = 200  # Sampling frequency in Hz
duration = len(ecg_signal) / fs  # Duration in seconds
time = np.linspace(0, duration, len(ecg_signal))

# Plot the ECG signal
plt.figure(figsize=(12, 6))
plt.plot(time, ecg_signal, color='b', linewidth=1)
plt.title('Raw ECG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.grid()
plt.show()
