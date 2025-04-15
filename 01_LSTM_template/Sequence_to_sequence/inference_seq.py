import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tqdm import tqdm  # for progress bar

# === CONFIGURATION ===
column_name = "sigma1eff[kN/m²]"
point_skip = 2
n_input = 20
n_output = 20  # sequence-to-sequence prediction length
data_known_percentage = 0.8
n_features = 1

# === PATH SETUP ===
folder_path = os.getcwd()
test_group_path = os.path.join(folder_path, "Test_Group_Multi")
file_name = "test_sample.txt"
result_folder = f"Results/test_sample/nsteps20_skips2_datapercent0.8_nbatch20"
saved_model_path = os.path.join(result_folder, "trained_model.keras")
output_path = os.path.join(result_folder, "inference_predicted_test_data.txt")

print("=== Sequence-to-Sequence Inference Starting ===")

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

# === SPLIT DATA ===
train_size = int(len(filtered_raw_seq) * data_known_percentage)
train_seq = filtered_raw_seq[:train_size]
test_seq = filtered_raw_seq[train_size:]
print(f"Train size: {len(train_seq)} | Test size: {len(test_seq)}")

# === SEQ2SEQ INFERENCE ===
print("[3/4] Running autoregressive seq2seq inference...")
predictions = []
x_input = train_seq[-n_input:]  # last window of training data

steps_needed = int(np.ceil(len(test_seq) / n_output))

for _ in tqdm(range(steps_needed), desc="Predicting blocks"):
    x_input_arr = np.array(x_input).reshape((1, n_input, n_features))
    yhat = model.predict(x_input_arr, verbose=0)[0]  # (n_output,)
    predictions.extend(yhat.tolist())
    x_input = x_input[n_output:] + yhat.tolist() if len(x_input) >= n_output else yhat.tolist()

# Trim to match test size
predictions = predictions[:len(test_seq)]

# === SAVE OUTPUT ===
print("[4/4] Saving predictions...")
df_predicted = pd.DataFrame({'Predicted Test': predictions})
df_predicted.to_csv(output_path, sep='\t', index=True)
print(f"Predictions saved to: {output_path}")
print("=== Inference Complete ===")