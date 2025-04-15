import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from sklearn.metrics import mean_squared_error

# === CONFIGURATION ===
result_folder = "Results/test_sample/nsteps20_skips2_datapercent0.8_nbatch20"
actual_path = os.path.join(result_folder, "actual_data.txt")
predicted_path = os.path.join(result_folder, "inference_predicted_test_data.txt")
plot_path = os.path.join(result_folder, "inference_vs_groundtruth.png")
metrics_path = os.path.join(result_folder, "inference_metrics.txt")

print("=== Evaluation Script Starting ===")

# === LOAD DATA ===
print("[1/4] Loading actual and predicted data...")
actual_df = pd.read_csv(actual_path, sep='\t')
predicted_df = pd.read_csv(predicted_path, sep='\t')

if 'Actual' not in actual_df.columns or 'Predicted Test' not in predicted_df.columns:
    raise KeyError("Expected columns not found in the input files.")

actual_data = actual_df['Actual'].values
predicted_data = predicted_df['Predicted Test'].values

# === TRIM TO LAST 20% OF ACTUAL DATA ===
print("[2/4] Extracting test portion of actual data...")
test_start_idx = int(0.8 * len(actual_data))
actual_test_data = actual_data[test_start_idx:]

if len(actual_test_data) != len(predicted_data):
    raise ValueError(f"Length mismatch: {len(actual_test_data)} actual vs {len(predicted_data)} predicted")

# === CALCULATE METRICS ===
print("[3/4] Calculating RMSE and Mean Percentage Error...")
mse = mean_squared_error(actual_test_data, predicted_data)
rmse = math.sqrt(mse)

with np.errstate(divide='ignore', invalid='ignore'):
    error_percent = np.abs((predicted_data - actual_test_data) / actual_test_data) * 100
    error_percent = np.nan_to_num(error_percent, nan=0.0, posinf=0.0, neginf=0.0)
mean_error_percent = np.mean(error_percent)

# === SAVE METRICS ===
with open(metrics_path, 'w') as f:
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"Mean Percentage Error: {mean_error_percent:.2f}%\n")
    f.write(f"Number of test points: {len(actual_test_data)}\n")

print(f"Saved evaluation metrics to: {metrics_path}")

# === PLOT RESULTS ===
print("[4/4] Plotting results...")
plt.figure(figsize=(12, 6))
plt.plot(actual_test_data, label='Ground Truth (Last 80%)', linewidth=2)
plt.plot(predicted_data, label='Predicted (Seq2Seq)', linestyle='--', linewidth=2)
plt.title("Seq2Seq Inference vs Ground Truth")
plt.xlabel("Time Step (relative)")
plt.ylabel("Stress (kN/m²)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_path)
plt.show()

print(f"Saved plot to: {plot_path}")
print("=== Evaluation Complete ===")
