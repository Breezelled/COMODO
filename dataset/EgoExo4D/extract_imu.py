import os
import pickle
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

dataset_dir = "your_dataset_path"


def resample_ns(
    signals: np.ndarray,
    timestamps_ns: np.ndarray,
    orig_rate: int,
    new_rate: int,
):
    """
    Resample the signals to a new rate.
    The function assumes timestamps are in nanoseconds.
    It converts timestamps to milliseconds for resampling,
    then converts the new timestamps back to nanoseconds.
    """
    # Convert nanoseconds to milliseconds
    timestamps_ms = timestamps_ns / 1e6
    signals_tensor = torch.as_tensor(signals)
    # Convert timestamps_ms to tensor of shape (N,1)
    ts_tensor = torch.from_numpy(timestamps_ms).unsqueeze(-1)

    # Resample the signals using torchaudio
    signals_resampled = torchaudio.functional.resample(
        waveform=signals_tensor, orig_freq=orig_rate, new_freq=new_rate
    ).numpy()

    # Number of samples after resampling
    nsamples = signals_resampled.shape[-1]
    period = 1 / new_rate  # period in seconds
    # Get initial time in seconds (from milliseconds)
    initial_sec = ts_tensor[0].item() / 1000.0
    # Generate new time instants (in seconds)
    ntimes = (torch.arange(nsamples, dtype=torch.float32) * period).view(
        -1, 1
    ) + initial_sec
    # Convert new timestamps to milliseconds and then to nanoseconds
    new_timestamps_ms = (ntimes * 1000).squeeze().numpy()
    new_timestamps_ns = new_timestamps_ms * 1e6
    return signals_resampled, new_timestamps_ns


def resampleIMU(signal, timestamps):
    """
    Given IMU signal and timestamps (timestamps in nanoseconds),
    compute the original sample rate and, if it is not 200Hz, resample
    the signal to 200Hz. Returns the resampled signal and new timestamps.
    """
    orig_rate = int(1e9 / np.mean(np.diff(timestamps)))
    print(f"Original sample rate: {orig_rate} Hz")
    if orig_rate != 200:
        signal_rs, timestamps_rs = resample_ns(signal, timestamps, orig_rate, 200)
        new_rate = int(1e9 / np.mean(np.diff(timestamps_rs)))
        print(f"Resampled sample rate: {new_rate} Hz")
        return signal_rs, timestamps_rs
    else:
        return signal, timestamps


def process_imu(folder):
    """
    For the given folder (which is a first-level subfolder under "takes"),
    load processed_imu.pkl, resample the IMU data to 200Hz,
    save the resampled data as processed_imu.npy in the same folder,
    and rename the original processed_imu.pkl to imu_1000hz.pkl.
    """
    root_path = dataset_dir + "takes"
    folder_path = os.path.join(root_path, folder)
    output_dir = os.path.join(dataset_dir, "processed_imu")
    imu_pkl = os.path.join(folder_path, "imu_1000hz.pkl")
    if not os.path.exists(imu_pkl):
        print(f"{imu_pkl} does not exist.")
        return

    # Load the pickle file (assumed to be a dict with keys "timestamps" and "data")
    with open(imu_pkl, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        timestamps = data.get("timestamps")
        sensor_data = data.get("data")
    else:
        print(f"Data format error in folder {folder}")
        return

    # Resample the IMU data to 200Hz (timestamps in ns)
    signal_rs, timestamps_rs = resampleIMU(sensor_data, timestamps)

    # Save the resampled data as an npy file in the same folder
    os.makedirs(output_dir, exist_ok=True)
    npy_file = os.path.join(output_dir, f"{folder}.npy")
    ts_file = os.path.join(output_dir, f"{folder}_timestamps.npy")

    # Save the resampled data and timestamps in separate files
    with open(npy_file, "wb") as f:
        np.save(f, signal_rs)
    with open(ts_file, "wb") as f:
        np.save(f, timestamps_rs)

    # Rename the original processed_imu.pkl to imu_1000hz.pkl
    print(f"Processed folder: {folder}")


def main():
    # Read folder names from folders_with_categories.txt (first column is the folder name)
    input_file = dataset_dir + "folders_with_categories.txt"
    folders = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                folders.append(parts[0])
    for folder in tqdm(folders):
        process_imu(folder)


if __name__ == "__main__":
    main()
