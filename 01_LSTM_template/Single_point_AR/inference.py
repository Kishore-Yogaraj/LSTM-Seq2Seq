import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tqdm import tqdm  # progress bar

# === CONFIGURATION ===
column_name = "sigma1eff[kN/m²]"
point_skip = 2
n_steps = 20
data_known_percentage = 0.5
n_features = 1

# Paths
folder_path = os.getcwd()
test_group_path = os.path.join(folder_path, "Test_Group_big")
file_name = "soil_sample_1.txt"
result_folder = f"Results/soil_sample_1/nsteps20_skips2_datapercent0.5_nbatch20"
saved_model_path = os.path.join(result_folder, "trained_model.keras")
output_path = os.path.join(result_folder, "inference_predicted_test_data.txt")

print("=== Inference Script Starting ===")

# === LOAD MODEL ===
print("[1/4] Loading model...")
if not os.path.exists(saved_model_path):
    raise FileNotFoundError(f"Trained model not found at: {saved_model_path}")
model = load_model(saved_model_path)
print(f"Model loaded from: {saved_model_path}")

# === LOAD DATA ===
print("[2/4] Loading input data...")
data_file = os.path.join(test_group_path, file_name)
if not os.path.exists(data_file):
    raise FileNotFoundError(f"Test data not found at: {data_file}")

data = pd.read_csv(data_file, sep='\t')
raw_seq = data[column_name].values.tolist()
filtered_raw_seq = [raw_seq[i] for i in range(len(raw_seq)) if i % point_skip == 0]
print(f"Loaded {len(filtered_raw_seq)} data points after point skip = {point_skip}")

# === SPLIT INTO TRAIN/TEST ===
train_size = int(len(filtered_raw_seq) * data_known_percentage)
train_seq = filtered_raw_seq[:train_size]
test_seq = filtered_raw_seq[train_size:]
print(f"Train size: {len(train_seq)} points | Test size: {len(test_seq)} points")

# === AUTOREGRESSIVE INFERENCE ===
print("[3/4] Running autoregressive inference...")
predictions = []
x_input = train_seq[-n_steps:]
x_input = np.array(x_input).reshape((1, n_steps, n_features))

for _ in tqdm(range(len(test_seq)), desc="Predicting"):
    yhat = model.predict(x_input, verbose=0)
    predictions.append(yhat[0][0])
    x_input = np.roll(x_input, -1)
    x_input[0, -1, 0] = yhat[0][0]

# === SAVE PREDICTIONS ===
print("[4/4] Saving predictions...")
df_predicted = pd.DataFrame({'Predicted Test': predictions})
df_predicted.to_csv(output_path, sep='\t', index=True)
print(f"Predictions saved to: {output_path}")
print("=== Inference Completed Successfully ===")
