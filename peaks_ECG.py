import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

# Load ECG data from a CSV file
file_path = './ecg.csv'
data = pd.read_csv(file_path)
first_data = data.head(200)

# Flatten the ECG data into a single array
ecg_signal = first_data.to_numpy().flatten()

# Create a time vector assuming a sampling frequency
fs = 200  # Sampling frequency in Hz
duration = len(ecg_signal) / fs  # Duration in seconds
time = np.linspace(0, duration, len(ecg_signal))

# Detect peaks in the ECG signal
peaks, _ = find_peaks(ecg_signal, height=0.5, distance=fs*0.5)  # Height threshold and min distance between peaks

# Plot the ECG signal with detected peaks
plt.figure(figsize=(12, 6))
plt.plot(time, ecg_signal, color='b', linewidth=1, label='ECG Signal')
plt.plot(time[peaks], ecg_signal[peaks], 'ro', label='Detected Peaks')
plt.title('ECG Signal with Detected Peaks')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.legend()
plt.grid()
plt.show()

# Analyze noise and trends
signal_mean = np.mean(ecg_signal)
signal_std = np.std(ecg_signal)
(signal_mean, signal_std, len(peaks))  # Return key stats and number of peaks for further analysis
print(f"Mean value of the signal: {signal_mean}")
print(f"Standard Deviaton of the signal: {signal_std}")