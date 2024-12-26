import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import norm

# Load the raw ECG signal data from the uploaded file
def load_ecg_data(file_path):
    import pandas as pd
    ecg_data = pd.read_csv(file_path)
    return ecg_data.to_numpy().flatten()

# Define a function to analyze peaks and compute discrete random variable statistics
def analyze_ecg_peaks(ecg_signal, fs=500):
    
    # Detect peaks in the ECG signal
    peaks, _ = find_peaks(ecg_signal, height=0.5, distance=fs*0.5)  # Height threshold and min distance between peaks
    
    # Total duration of the signal in seconds
    total_duration = len(ecg_signal) / fs
    
    # Divide the signal into 1-second intervals and count peaks per interval
    num_intervals = int(total_duration)
    peaks_per_second = np.zeros(num_intervals)
    for i in range(num_intervals):
        start_idx = i * fs
        end_idx = (i + 1) * fs
        peaks_in_interval = np.sum((peaks >= start_idx) & (peaks < end_idx))
        peaks_per_second[i] = peaks_in_interval
    
    # Define the discrete random variable X: number of peaks per second
    unique_values, counts = np.unique(peaks_per_second, return_counts=True)
    prob_distribution = counts / num_intervals
    
    # Calculate expected value, variance, and standard deviation
    expected_value = np.sum(unique_values * prob_distribution)
    variance = np.sum(((unique_values - expected_value) ** 2) * prob_distribution)
    std_dev = np.sqrt(variance)
    
    # Print results
    print("Discrete Random Variable: Number of Peaks per Second")
    print("Probability Distribution:")
    for val, prob in zip(unique_values, prob_distribution):
        print(f"P(X = {int(val)}) = {prob:.4f}")
    print(f"\nExpected Value (E[X]): {expected_value:.2f}")
    print(f"Variance (Var[X]): {variance:.2f}")
    print(f"Standard Deviation (sigma): {std_dev:.2f}")
    
    # Plot the probability distribution
    plt.figure(figsize=(10, 6))
    plt.bar(unique_values, prob_distribution, color='c', edgecolor='k')
    plt.title('Probability Distribution of Peaks per Second')
    plt.xlabel('Number of Peaks per Second')
    plt.ylabel('Probability')
    plt.grid(axis='y')
    plt.show()
    
    return prob_distribution, expected_value, variance, std_dev

# Define a function to analyze a continuous random variable from the ECG signal
def analyze_ecg_amplitude(ecg_signal):
    
    # Calculate statistics for the ECG signal amplitude
    mean = np.mean(ecg_signal)
    variance = np.var(ecg_signal)
    std_dev = np.std(ecg_signal)
    
    # Generate a normal distribution for the amplitude
    ecg_min = np.min(ecg_signal)
    ecg_max = np.max(ecg_signal)
    x = np.linspace(ecg_min, ecg_max, 1000)
    pdf = norm.pdf(x, mean, std_dev)
    
    # Print results
    print("Continuous Random Variable: ECG Signal Amplitude")
    print(f"Expected Value (Mean): {mean:.4f}")
    print(f"Variance: {variance:.4f}")
    print(f"Standard Deviation: {std_dev:.4f}")
    
    # Plot the probability distribution
    plt.figure(figsize=(10, 6))
    plt.hist(ecg_signal, bins=50, density=True, color='lightblue', edgecolor='k', label='ECG Amplitude Histogram')
    plt.plot(x, pdf, 'r', label='Fitted Normal Distribution')
    plt.title('Probability Distribution of ECG Signal Amplitude')
    plt.xlabel('Amplitude (mV)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid()
    plt.show()
    
    return mean, variance, std_dev

# Main execution
if __name__ == "__main__":
    
    # Load ECG data
    ecg_signal = load_ecg_data("./ecg.csv")
    
    # Analyze the signal and compute discrete random variable statistics
    analyze_ecg_peaks(ecg_signal, fs=500)
    
    # Analyze the ECG amplitude as a continuous random variable
    analyze_ecg_amplitude(ecg_signal)