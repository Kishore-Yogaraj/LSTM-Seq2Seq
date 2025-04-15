import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from sklearn.metrics import mean_squared_error

# === CONFIGURATION ===
result_folder = "Results/soil_sample_1/nsteps20_skips2_datapercent0.5_nbatch20"
actual_path = os.path.join(result_folder, "actual_data.txt")
predicted_path = os.path.join(result_folder, "inference_predicted_test_data.txt")
plot_path = os.path.join(result_folder, "inference_vs_groundtruth.png")
metrics_path = os.path.join(result_folder, "inference_metrics.txt")

# === LOAD DATA ===
actual_data = pd.read_csv(actual_path, sep='\t')['Actual'].values
predicted_data = pd.read_csv(predicted_path, sep='\t')['Predicted Test'].values

# === TRIM ACTUAL DATA TO LAST X% ===
test_start_idx = int(0.5 * len(actual_data))
actual_test_data = actual_data[test_start_idx:]

# === SANITY CHECK ===
if len(actual_test_data) != len(predicted_data):
    raise ValueError(f"Length mismatch: {len(actual_test_data)} actual vs {len(predicted_data)} predicted")

# === CALCULATE METRICS ===
mse = mean_squared_error(actual_test_data, predicted_data)
rmse = math.sqrt(mse)

# Error % per point
error_percent = np.abs((predicted_data - actual_test_data) / actual_test_data) * 100
mean_error_percent = np.mean(error_percent)

# === SAVE METRICS TO FILE ===
with open(metrics_path, 'w') as f:
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"Mean Percentage Error: {mean_error_percent:.2f}%\n")
    f.write(f"Number of test points: {len(actual_test_data)}\n")

print(f"Saved evaluation metrics to: {metrics_path}")

# === PLOT ===
plt.figure(figsize=(12, 6))
plt.plot(actual_test_data, label='Ground Truth (Last 50%)')
plt.plot(predicted_data, label='Predicted', linestyle='--')
plt.title("Inference Prediction vs Ground Truth")
plt.xlabel("Time Step")
plt.ylabel("Stress (kN/m²)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_path)
plt.show()

print(f"Saved plot to: {plot_path}")
