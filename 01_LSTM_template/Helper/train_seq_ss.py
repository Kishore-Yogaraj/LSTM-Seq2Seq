import pandas as pd
import os
import numpy as np
import shutil
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from sklearn.metrics import mean_squared_error
import math
from tensorflow.keras.callbacks import EarlyStopping
import time
import statistics
from tensorflow.keras.backend import clear_session
from multiprocessing import Pool
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError

seed = 1993
tf.random.set_seed(seed)
np.random.seed(seed)

# === Force usage of GPU if available ===
if tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
        print("Using GPU for training.")
    except Exception as e:
        print("Could not set memory growth for GPU:", e)
else:
    print("No GPU found. Training will proceed on CPU.")

data_known_percentage = [0.8]
point_skip = [2]
n_steps = [20]
n_batches = [20]
column_name = "sigma1eff[kN/m²]"
folder_path = os.getcwd()
n_features = 1
group_size = 1
patience = 500
n_epochs = 500
central_save_loc = f'{folder_path}/Results'
test_group_path = f'{folder_path}/Test_Group_big'

def split_sequence(sequence, n_input, n_output):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_input
        out_end_ix = end_ix + n_output
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def save_loss_rmse(losses, rmses, second_layer_dir):
    df_loss_rmse = pd.DataFrame({
        'Epoch': range(1, len(losses) + 1),
        'Loss': losses,
        'RMSE': rmses
    })
    df_loss_rmse.to_csv(f'{second_layer_dir}/loss_rmse_epochs.txt', index=False, sep='\t')

def calculate_rmse(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return math.sqrt(mse)

def error_calc(pred, test):
    error_abs = np.abs(np.subtract(pred, test))
    error = [(error_abs[x] / np.abs(test[x])) * 100 for x in range(len(error_abs))]
    error_mean = statistics.mean(error)
    print(f'The error mean is {error_mean}')
    return error, error_mean

def check_for_nan(array, array_name):
    if np.isnan(array).any():
        print(f"NaN values found in {array_name} at indices: {np.where(np.isnan(array))}")
    else:
        print(f"No NaN values found in {array_name}")
    return np.isnan(array).all()

def process_file(file):
    data = pd.read_csv(os.path.join(test_group_path, file), sep='\t', header=0)
    raw_seq = data[column_name].values.tolist()
    file_base_name = os.path.splitext(file)[0]
    first_layer_dir = os.path.join(central_save_loc, file_base_name)
    if os.path.exists(first_layer_dir):
        shutil.rmtree(first_layer_dir)
    os.makedirs(first_layer_dir)

    for kp in data_known_percentage:
        for ps in point_skip:
            filtered_raw_seq = [raw_seq[raw_i] for raw_i in range(len(raw_seq)) if raw_i % ps == 0]
            for step in n_steps:
                for batch in n_batches:
                    process_configuration(first_layer_dir, filtered_raw_seq, kp, ps, step, batch)

def process_configuration(first_layer_dir, filtered_raw_seq, kp, ps, step, batch):
    train_size = int(len(filtered_raw_seq) * kp)
    train_seq, test_seq = filtered_raw_seq[:train_size], filtered_raw_seq[train_size:]
    n_output = step
    X_train, y_train = split_sequence(train_seq, step, n_output)
    X_train = X_train.reshape((X_train.shape[0], step, n_features))
    y_train = y_train.reshape((y_train.shape[0], n_output))

    result_folder_name = f'nsteps{step}_skips{ps}_datapercent{kp}_nbatch{batch}'
    second_layer_dir = os.path.join(first_layer_dir, result_folder_name)
    if os.path.exists(second_layer_dir):
        shutil.rmtree(second_layer_dir)
    os.makedirs(second_layer_dir)

    model = Sequential()
    model.add(LSTM(units=25, return_sequences=True, input_shape=(step, n_features)))
    model.add(LSTM(units=25, return_sequences=True))
    model.add(LSTM(units=25, return_sequences=False))
    model.add(Dense(step, activation='linear'))

    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.MeanSquaredError()
    train_loss_results = []
    train_rmse_results = []

    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch)

    @tf.function
    def train_step(x, y, epoch):
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            use_ground_truth = tf.random.uniform([batch, step], 0, 1) < tf.cast(tf.maximum(1 - epoch / n_epochs, 0.1), tf.float32)
            mixed_targets = tf.where(use_ground_truth, tf.cast(y, tf.float32), predictions)
            loss = loss_fn(mixed_targets, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        rmse = tf.sqrt(loss_fn(tf.cast(y, tf.float32), predictions))
        return loss, rmse

    # === Training Loop ===
    start_time = time.time()
    for epoch in range(n_epochs):
        batch_losses = []
        batch_rmses = []
        for x_batch, y_batch in dataset:
            loss, rmse = train_step(x_batch, y_batch, epoch)
            batch_losses.append(loss.numpy())
            batch_rmses.append(rmse.numpy())
        epoch_loss = np.mean(batch_losses)
        epoch_rmse = np.mean(batch_rmses)
        train_loss_results.append(epoch_loss)
        train_rmse_results.append(epoch_rmse)
        print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}, RMSE: {epoch_rmse:.4f}")

    end_time = time.time()

    saved_model_path = os.path.join(second_layer_dir, "trained_model")  # No extension
    model.save(saved_model_path, save_format='tf')  # Explicitly use SavedModel format
    print(f"Model saved in TensorFlow SavedModel format at: {second_layer_dir}")

    save_loss_rmse(train_loss_results, train_rmse_results, second_layer_dir)

    clear_session()

if __name__ == '__main__':
    files = [file for file in os.listdir(test_group_path) if file.endswith(".txt")]
    for file in files:
        process_file(file)
