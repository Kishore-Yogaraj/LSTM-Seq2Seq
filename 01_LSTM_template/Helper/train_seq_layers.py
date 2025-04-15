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

# Check if GPU is available
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#
# # Set the GPU device (uncomment if using GPU)
# gpu_devices = tf.config.list_physical_devices('GPU')
# if gpu_devices:
#     tf.config.experimental.set_memory_growth(gpu_devices[0], True)

seed = 1993
tf.random.set_seed(seed)
np.random.seed(seed)

# data_known_percentage = [0.6, 0.7, 0.8]
data_known_percentage = [0.8]
# point_skip = [1, 2, 3]
point_skip = [2]
# n_steps = [10, 20, 50]
n_steps = [20]
# n_batches = [5, 10, 20, 50]
n_batches = [20]
column_name = "sigma1eff[kN/m²]"
folder_path = os.getcwd()
n_features = 1
group_size = 1
patience = 500
n_epochs = 5000
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


def save_loss_rmse(hist, second_layer_dir):
    df_loss_rmse = pd.DataFrame({
        'Epoch': range(1, len(hist.history['loss']) + 1),
        'Loss': hist.history['loss'],
        'RMSE': hist.history['root_mean_squared_error']
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
    model.add(LSTM(units=25, activation='relu', return_sequences=True, input_shape=(step, n_features)))
    model.add(LSTM(units=25, activation='relu', return_sequences=True))
    model.add(LSTM(units=25, activation='relu', return_sequences=False))
    model.add(Dense(step, activation='linear'))
    model.compile(
        optimizer='adam',
        loss=MeanSquaredError(),
        metrics=[RootMeanSquaredError()]
    )

    early_stopping = EarlyStopping(
        monitor='loss',
        min_delta=1,
        patience=patience,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )

    start_time = time.time()
    history = model.fit(
        X_train,
        y_train,
        epochs=n_epochs,
        verbose=2,
        batch_size=batch,
        callbacks=[early_stopping]
    )
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Total training time: {runtime:.2f} seconds")
    

    with open(f'{second_layer_dir}/model_summary.txt', 'w', encoding='utf-8') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # === Save model using SavedModel format ===
    saved_model_path = os.path.join(second_layer_dir, "trained_model.keras")
    model.save(saved_model_path)
    print(f"Model saved in TensorFlow SavedModel format at: {saved_model_path}")

    # === Autoregressive training prediction only ===
    predicted_train = []
    x_input_train = train_seq[:step]
    x_input_train = np.array(x_input_train).reshape((1, step, n_features))
    for m in range(len(train_seq) - step):
        yhat_train = model.predict(x_input_train, verbose=0)
        predicted_train.append(yhat_train[0][0])
        x_input_train = np.roll(x_input_train, -1)
        x_input_train[0, -1, 0] = yhat_train[0][0]

    allNan_train = check_for_nan(np.array(train_seq[step:]), 'train_seq[step:]')
    allNan_predicted = check_for_nan(np.array(predicted_train), 'predicted_train')
    train_seq_clean = np.array(train_seq[step:])
    train_seq_clean = train_seq_clean[~np.isnan(train_seq_clean)]
    predicted_train_clean = np.array(predicted_train)
    predicted_train_clean = predicted_train_clean[~np.isnan(predicted_train_clean)]

    # === Skip test prediction and fill placeholder instead ===
    predictions = []  # empty test predictions
    perc_mean = float('nan')  # or 0.0 if you prefer to avoid NaN

    if not allNan_train or not allNan_predicted:
        min_length = min(len(train_seq_clean), len(predicted_train_clean))
        train_seq_clean = train_seq_clean[:min_length]
        predicted_train_clean = predicted_train_clean[:min_length]

        df_actual = pd.DataFrame({'Actual': filtered_raw_seq[:]})
        df_actual.to_csv(f'{second_layer_dir}/actual_data.txt', index=True, sep='\t')

        df_predicted_train = pd.DataFrame({'Predicted Training': predicted_train})
        df_predicted_train.to_csv(f'{second_layer_dir}/predicted_train_data.txt', index=True, sep='\t')

        # Placeholder for skipped test prediction
        df_predicted_test = pd.DataFrame({'Predicted Test': predictions})
        df_predicted_test.to_csv(f'{second_layer_dir}/predicted_test_data.txt', index=True, sep='\t')

        df_error_percent = pd.DataFrame({'Error': []})
        df_error_percent.to_csv(f'{second_layer_dir}/error_percent.txt', index=True, sep='\t')

        runtime = end_time - start_time
        num_calculated_epoch = len(history.history['loss'])

        with open(f'{second_layer_dir}/log_file.txt', 'w') as log_file:
            log_file.write(f"Time of running: {runtime} seconds\n")
            log_file.write(f"Number of epochs set: {n_epochs}\n")
            log_file.write(f"Number of epochs calculated: {num_calculated_epoch}\n")
            log_file.write(f"Best epoch chosen at end: {num_calculated_epoch - patience}\n")
            log_file.write(f"Final loss: {history.history['loss'][-1]}\n")
            log_file.write(f"Final RMSE: {history.history['root_mean_squared_error'][-1]}\n")
            log_file.write(f"Length of actual data: {len(filtered_raw_seq)}\n")
            log_file.write(f"Length of predicted training: {len(predicted_train)}\n")
            log_file.write(f"Length of predicted test: {len(predictions)}\n")
            log_file.write(f"Mean Error Percentage: {perc_mean}\n")

        save_loss_rmse(history, second_layer_dir)
    else:
        print(f'All values were returned as NaN (not a number). No calculations possible')
        with open(f'{second_layer_dir}/log_file.txt', 'w') as log_file:
            log_file.write(f"Time of running: {end_time - start_time} seconds\n")
            log_file.write(f"All values returned as NaN (not a number). No calculations possible")
    clear_session()

if __name__ == '__main__':
    files = [file for file in os.listdir(test_group_path) if file.endswith(".txt")]

    for file in files:
        process_file(file)
