dataset_dir = "your_dataset_path"
output_file_name = "processed_imu.pkl"

import os
import pickle
import numpy as np
import pyvrs
import glob
import multiprocessing as mp
import tqdm

def read_imu_vrs(vrs_path):
    vrs_reader = pyvrs.reader.SyncVRSReader(vrs_path)

    invalid_record_count = 0
    
    imu_reader = vrs_reader.filtered_by_fields(
        stream_ids='1202-1', # 1202-2 for 800 Hz, 1202-1 for 1000 Hz 
        record_types='data'
    )
    
    timestamps = []
    acc_data = []
    gyro_data = []
    
    for record in imu_reader:
        meta_block = record.metadata_blocks[0]
        if meta_block['accelerometer_valid'] and meta_block['gyroscope_valid']:
            timestamps.append(meta_block['capture_timestamp_ns'])
            acc_data.append(meta_block['accelerometer'])
            gyro_data.append(meta_block['gyroscope'])
        else:
            invalid_record_count += 1
            # print(f"t: {meta_block['capture_timestamp_ns']}, acc_valid: {meta_block['accelerometer_valid']} gyro_valid: {meta_block['gyroscope_valid']}")

    vrs_reader.close()
    
    timestamps, acc_data, gyro_data = np.array(timestamps), np.array(acc_data), np.array(gyro_data)
    sensor_data = np.concatenate([acc_data, gyro_data], axis=1).transpose()

    return timestamps, sensor_data, invalid_record_count


def f(take_dir):
    # get vrs files which contains IMU traces

    all_vrs = glob.glob(os.path.join(take_dir, "*_noimagestreams.vrs"))
    # if len(all_vrs) != 1:
    #     take_dirs_with_odd_imu.append((os.path.basename(take_dir), len(all_vrs)))
    if len(all_vrs) > 0:
        vrs = all_vrs[0]

        out_path = os.path.join(take_dir, output_file_name)
        original_relative_path = os.path.relpath(vrs, dataset_dir)
        
        timestamps, sensor_data, invalid_record_count = read_imu_vrs(vrs)
        
        with open(out_path, "wb") as f:
            pickle.dump({
                "timestamps": timestamps,
                "data": sensor_data,
                "invalid_record_count": invalid_record_count,
                "original_relative_path": original_relative_path,
            }, f)

        print(f"Written to {out_path}")

take_dirs = sorted(glob.glob(os.path.join(dataset_dir, "takes", "*")))
print(len(take_dirs))

with mp.Pool(processes=128) as pool:
    tasks = take_dirs
    for _ in tqdm.tqdm(pool.imap_unordered(f, tasks), total=len(tasks)):
        pass

print("Finished")