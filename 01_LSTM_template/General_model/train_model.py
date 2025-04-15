import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.backend import clear_session

# === Configuration ===
folder_path = os.getcwd()
data_folder = os.path.join(folder_path, "Train_Group_Multi")
save_dir = os.path.join(folder_path, "Generalized_Results")
column_name = "sigma1eff[kN/m²]"
n_input = 20
n_output = 20
n_features = 1
batch_size = 32
n_epochs = 300
patience = 20
seed = 1993
tf.random.set_seed(seed)
np.random.seed(seed)

# === Utility: split sequence into input-output pairs ===
def split_sequence(sequence, n_input, n_output):
    X, y = [], []
    for i in range(len(sequence) - n_input - n_output + 1):
        seq_x = sequence[i:i + n_input]
        seq_y = sequence[i + n_input:i + n_input + n_output]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# === Collect all training data from all files ===
X_all, y_all = [], []

print("Loading data from files...")
for filename in os.listdir(data_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(data_folder, filename)
        try:
            df = pd.read_csv(file_path, sep='\t')
            if column_name not in df.columns:
                print(f"Column '{column_name}' missing in {filename}. Skipping.")
                continue

            raw_seq = df[column_name].dropna().values
            X_file, y_file = split_sequence(raw_seq, n_input, n_output)
            X_all.append(X_file)
            y_all.append(y_file)
            print(f"Loaded {filename}: {X_file.shape[0]} samples")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# === Final training set ===
X_train = np.concatenate(X_all, axis=0).reshape(-1, n_input, n_features)
y_train = np.concatenate(y_all, axis=0)
print(f"Total training samples: {X_train.shape[0]}")

# === Define model ===
model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(n_input, n_features)),
    LSTM(64, activation='relu'),
    Dense(n_output)
])
model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=[RootMeanSquaredError()])

# === Callbacks ===
early_stopping = EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)

# === Train model ===
history = model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, verbose=2, callbacks=[early_stopping])

# === Save results ===
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model.save(os.path.join(save_dir, "generalized_lstm_model.keras"))

pd.DataFrame({
    "Epoch": range(1, len(history.history["loss"]) + 1),
    "Loss": history.history["loss"],
    "RMSE": history.history["root_mean_squared_error"]
}).to_csv(os.path.join(save_dir, "training_log.txt"), sep='\t', index=False)

print(f"Model and training log saved to {save_dir}")
clear_session()
