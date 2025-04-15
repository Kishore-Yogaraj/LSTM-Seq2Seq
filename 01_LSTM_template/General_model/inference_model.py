import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math

# === CONFIGURATION ===
column_name = "sigma1eff[kN/m²]"
point_skip = 2
n_input = 20
n_output = 20
data_known_percentage = 0.8
n_features = 1

# === PATH SETUP ===
folder_path = os.getcwd()
data_folder = os.path.join(folder_path, "Test_Group_Multi")
test_file = "test_sample.txt"
test_path = os.path.join(data_folder, test_file)
model_path = os.path.join(folder_path, "Generalized_Results", "generalized_lstm_model.keras")

# === OUTPUT FILES ===
inference_folder = os.path.join(folder_path, "Inference_Results")
os.makedirs(inference_folder, exist_ok=True)

full_filtered_path = os.path.join(inference_folder, "filtered_test_data.txt")
known_data_path = os.path.join(inference_folder, "inference_input_80_percent.txt")
predicted_data_path = os.path.join(inference_folder, "inference_predicted_20_percent.txt")

# === STEP 1: LOAD & FILTER TEST DATA ===
print("[1/5] Loading and filtering test sample...")
if not os.path.exists(test_path):
    raise FileNotFoundError(f"Test file not found at: {test_path}")

df = pd.read_csv(test_path, sep='\t')
if column_name not in df.columns:
    raise ValueError(f"Column '{column_name}' not found in test sample.")

raw_seq = df[column_name].dropna().values
filtered_seq = [raw_seq[i] for i in range(len(raw_seq)) if i % point_skip == 0]

# Save filtered version for reference
pd.DataFrame({'Filtered Stress': filtered_seq}).to_csv(full_filtered_path, sep='\t', index=False)
print(f"Saved full filtered sequence to: {full_filtered_path}")

# === STEP 2: Split into known and ground truth (NO NORMALIZATION) ===
split_index = int(len(filtered_seq) * data_known_percentage)
known_input = filtered_seq[:split_index]
ground_truth = filtered_seq[split_index:]

# Save known input
pd.DataFrame({'Known Input': known_input}).to_csv(known_data_path, sep='\t', index=False)
print(f"Saved 80% known input to: {known_data_path}")

# === STEP 3: Load Model ===
print("[2/5] Loading trained model...")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Trained model not found at: {model_path}")
model = load_model(model_path)
print(f"Model loaded from: {model_path}")

# === STEP 4: Run Autoregressive Inference ===
print("[3/5] Starting autoregressive prediction...")
predictions = []
x_input = known_input[-n_input:].copy()

steps_needed = int(np.ceil(len(ground_truth) / n_output))
for _ in tqdm(range(steps_needed), desc="Predicting blocks"):
    x_input_arr = np.array(x_input[-n_input:]).reshape((1, n_input, n_features))
    yhat = model.predict(x_input_arr, verbose=0)[0]  # (n_output,)
    predictions.extend(yhat.tolist())
    x_input.extend(yhat.tolist())

# Trim predictions to match ground truth
predictions = predictions[:len(ground_truth)]

# Save predictions
pd.DataFrame({'Predicted Stress': predictions}).to_csv(predicted_data_path, sep='\t', index=False)
print(f"Saved predictions to: {predicted_data_path}")

# === STEP 5: Plotting ===
print("[4/5] Plotting and saving results...")
plt.figure(figsize=(10, 4))
plt.plot(range(len(filtered_seq)), filtered_seq, label="Original Full Sequence", color='gray', alpha=0.4)
plt.plot(range(split_index, split_index + len(predictions)), predictions, label="Predicted", color='red')
plt.plot(range(split_index, split_index + len(ground_truth)), ground_truth, label="Ground Truth", color='green')
plt.title("Predicted vs. Actual Stress Curve (Last 20%)")
plt.xlabel("Time Step")
plt.ylabel("Stress (kN/m²)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot
plot_path = os.path.join(inference_folder, "prediction_plot.png")
plt.savefig(plot_path)
plt.close()
print(f"Plot saved to: {plot_path}")

# === STEP 6: Compute Error Metrics ===
print("[5/5] Computing error metrics...")
actual = np.array(ground_truth[:len(predictions)])
predicted = np.array(predictions)

rmse = math.sqrt(mean_squared_error(actual, predicted))
percent_errors = np.abs((actual - predicted) / (np.abs(actual) + 1e-8)) * 100
mean_percent_error = np.mean(percent_errors)

# Save metrics
metrics_path = os.path.join(inference_folder, "metrics.txt")
with open(metrics_path, "w") as f:
    f.write(f"RMSE: {rmse:.4f} kN/m²\n")
    f.write(f"Mean Percent Error: {mean_percent_error:.2f}%\n")

print(f"Saved error metrics to: {metrics_path}")

# Save comparison
comparison_df = pd.DataFrame({
    "Actual Stress": actual,
    "Predicted Stress": predicted,
    "Percent Error": percent_errors
})
comparison_path = os.path.join(inference_folder, "actual_vs_predicted.txt")
comparison_df.to_csv(comparison_path, sep='\t', index=False)
print(f"Saved actual vs. predicted comparison to: {comparison_path}")

print("=== Inference Complete ===")
