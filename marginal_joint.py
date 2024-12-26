import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Function to load signals from CSV files
def load_signal1(file_path):
    data = pd.read_csv(file_path)
    f_data = data.head(100)
    return data.iloc[:, 0].values  # Extracting the first column


def load_signal2(file_path):
    data = pd.read_csv(file_path)
    f_data = data.head(200)
    return f_data.iloc[:, 0].values  # Extracting the first column

# File paths for the two biomedical signals
file_path_1 = './ecg.csv'  
file_path_2 = './ecg.csv' 

# Load the signals from CSV files
signal_1 = load_signal1(file_path_1)
signal_2 = load_signal2(file_path_2)

# 1. Marginal Probability Distributions (Histograms)
plt.figure(figsize=(12, 6))

# Signal 1
plt.subplot(1, 2, 1)
plt.hist(signal_1, bins=30, density=True, alpha=0.6, color='g')
plt.title('Marginal Probability Distribution of Signal 1')
plt.xlabel('Signal 1')
plt.ylabel('Probability Density')

# Signal 2
plt.subplot(1, 2, 2)
plt.hist(signal_2, bins=30, density=True, alpha=0.6, color='b')
plt.title('Marginal Probability Distribution of Signal 2')
plt.xlabel('Signal 2')
plt.ylabel('Probability Density')

plt.tight_layout()
plt.show()

# 2. Covariance
covariance = np.cov(signal_1, signal_2)[0, 1]
print(f"Covariance between Signal 1 and Signal 2: {covariance}")

# 3. Correlation Coefficient
correlation_coefficient = np.corrcoef(signal_1, signal_2)[0, 1]
print(f"Correlation Coefficient between Signal 1 and Signal 2: {correlation_coefficient}")
