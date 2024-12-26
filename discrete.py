import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load the raw ECG signal data from the uploaded file
def load_ecg_data(file_path):
    import pandas as pd
    ecg_data = pd.read_csv(file_path)
    return ecg_data.to_numpy().flatten()

# Define a function to analyze peaks and compute discrete random variable statistics
def analyze_ecg_peaks(ecg_signal, fs=200):
    """
    Analyze the ECG signal to define a discrete random variable (number of peaks per second),
    and calculate its probability distribution, expected value, variance, and standard deviation.
    
    Parameters:
    - ecg_signal: The raw ECG signal data.
    - fs: Sampling frequency (Hz).
    
    Returns:
    - prob_distribution: Probability distribution of the discrete random variable.
    - expected_value: Expected value of the random variable.
    - variance: Variance of the random variable.
    - std_dev: Standard deviation of the random variable.
    """
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

# Main execution
if __name__ == "__main__":
    # File path to the uploaded ECG data
    file_path = './ecg.csv'
    
    # Load ECG data
    ecg_signal = load_ecg_data(file_path)
    
    # Analyze the signal and compute statistics
    analyze_ecg_peaks(ecg_signal, fs=500)