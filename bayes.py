import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Step 1: Load the ECG data
def load_ecg_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Step 2: Preprocess the ECG signal (assuming it's in the first column)
def preprocess_ecg_data(data):
    # Extract the first column (assuming ECG signal is in the first column)
    ecg_signal = data.iloc[:, 0].values  # Get the first column as ECG signal
    return ecg_signal

# Step 3: Compute basic probabilities (e.g., P(heartbeat), P(noise))
def compute_basic_probabilities(ecg_signal):
    # Example: Detecting a heartbeat (based on some threshold or pattern)
    threshold = np.mean(ecg_signal) + 2 * np.std(ecg_signal)  # Example threshold for heartbeat detection
    heartbeat_detected = ecg_signal > threshold
    P_heartbeat = np.sum(heartbeat_detected) / len(ecg_signal)
    
    # Example: Detecting noise (based on variance)
    noise_detected = np.abs(np.diff(ecg_signal)) > np.std(ecg_signal)  # Simple noise detection
    noise_detected = np.append(noise_detected, False)  # Padding to match the length
    P_noise = np.sum(noise_detected) / len(ecg_signal)
    
    return P_heartbeat, P_noise

# Step 4: Apply Bayes' Theorem (P(A|B) = P(B|A) * P(A) / P(B))
def bayesian_update(P_A, P_B_given_A, P_B):
    # Bayes' Theorem
    P_A_given_B = (P_B_given_A * P_A) / P_B
    return P_A_given_B

# Step 5: Conditional Probability (P(A|B) = P(A and B) / P(B))
def conditional_probability(event_A, event_B):
    P_A_and_B = np.sum(event_A & event_B) / len(event_A)
    P_B = np.sum(event_B) / len(event_B)
    P_A_given_B = P_A_and_B / P_B
    return P_A_given_B

# Step 6: Check Independence (P(A and B) = P(A) * P(B) for independence)
def check_independence(event_A, event_B):
    P_A = np.sum(event_A) / len(event_A)
    P_B = np.sum(event_B) / len(event_B)
    P_A_and_B = np.sum(event_A & event_B) / len(event_A)
    
    if np.isclose(P_A_and_B, P_A * P_B):
        return True  # Events are independent
    else:
        return False  # Events are dependent

# Step 7: Visualize the ECG signal and detected events
def visualize_ecg(ecg_signal, heartbeat_detected, noise_detected):
    plt.figure(figsize=(10, 6))
    plt.plot(ecg_signal, label='ECG Signal')
    plt.plot(np.where(heartbeat_detected)[0], ecg_signal[heartbeat_detected], 'ro', label='Heartbeat Detected')
    plt.plot(np.where(noise_detected)[0], ecg_signal[noise_detected], 'go', label='Noise Detected')
    plt.legend()
    plt.title('ECG Signal with Heartbeat and Noise Detection')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

# Main function to analyze the ECG signal
def analyze_ecg(file_path):
    data = load_ecg_data(file_path)
    ecg_signal = preprocess_ecg_data(data)
    
    # Compute basic probabilities
    P_heartbeat, P_noise = compute_basic_probabilities(ecg_signal)
    print(f"P(Heartbeat): {P_heartbeat}")
    print(f"P(Noise): {P_noise}")
    
    # Example of applying Bayes' Theorem (update probability based on new evidence)
    P_B_given_A = 0.8  # Example: Probability of noise given a heartbeat
    P_B = P_noise  # P(Noise)
    P_A = P_heartbeat  # P(Heartbeat)
    P_A_given_B = bayesian_update(P_A, P_B_given_A, P_B)
    print(f"P(Heartbeat|Noise): {P_A_given_B}")
    
    # Example of conditional probability
    heartbeat_detected = ecg_signal > np.mean(ecg_signal) + 2 * np.std(ecg_signal)
    noise_detected = np.abs(np.diff(ecg_signal)) > np.std(ecg_signal)
    noise_detected = np.append(noise_detected, False)  # Padding to match the length
    P_heartbeat_given_noise = conditional_probability(heartbeat_detected, noise_detected)
    print(f"P(Heartbeat|Noise): {P_heartbeat_given_noise}")
    
    # Check if heartbeat and noise are independent
    independence = check_independence(heartbeat_detected, noise_detected)
    print(f"Are Heartbeat and Noise Independent? {independence}")
    
    # Visualize the ECG signal with detections
    visualize_ecg(ecg_signal, heartbeat_detected, noise_detected)

# Execution
file_path = './ecg.csv'
analyze_ecg(file_path)
