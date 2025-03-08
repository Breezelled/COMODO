from torchvision.transforms import (
    Compose,
)
from .collate_util import *
import torch
from pytorchvideo.data.video import VideoPathHandler
import pandas as pd
import numpy as np

n_channels = 6


class IMUDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_name,
        imu_path,
        label2id,
        dataset_name: str,
        imu_is_raw: bool = True,
    ):
        self.imu_path = imu_path
        self.imu_is_raw = imu_is_raw
        self.label2id = label2id
        self.Racc = 16384
        self.Rgyro = 16.4
        self.dataset_name = dataset_name

        with open(file_name, "r") as f:
            lines = f.readlines()

        self.data = []
        for line in lines:
            self.data.extend(self._process_line(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"imu": self.data[idx]["imu"], "label": self.data[idx]["label"]}

    def _process_line_uestc(self, line):
        line = line.strip()
        original_path, *_ = line.split()
        original_path = original_path.replace("yourdataset_path/data/", "")
        imu_path = (self.imu_path / original_path).with_suffix(".csv")
        class_label = imu_path.parent.name.split("_", 1)[1]

        df = pd.read_csv(imu_path)
        if self.imu_is_raw:
            accel_data = df.iloc[:, :3] / self.Racc
            gyro_data = df.iloc[:, 3:6] / self.Rgyro
            imu_data = pd.concat([accel_data, gyro_data], axis=1).values
        else:
            imu_data = df.iloc[:, :6].values
        
        imu_data = np.transpose(imu_data, (1, 0)) # [num_channels, seq_len]

        return [
            {
                "imu": torch.tensor(imu_data, dtype=torch.float32),
                "label": self.label2id[class_label],
            }
        ]

    def _process_line_ego4d(self, line):
        uid, scenario = line.strip().split("\t")
        scenario = scenario.strip()
        imu_path = self.imu_path / f"{uid}.npy"

        # [num_classes, seq_len] -> [seq_len, num_classes]
        imu_data = np.load(imu_path).T
        imu_data = torch.tensor(imu_data, dtype=torch.float32)

        steps_per_clip = int(5 * 200)
        num_clips = imu_data.shape[0] // steps_per_clip

        data_samples = []
        for i in range(num_clips):
            start = i * steps_per_clip
            end = start + steps_per_clip

            if end > imu_data.shape[0]:
                break

            clip = imu_data[start:end, :]
            assert clip.shape[0] == steps_per_clip

            clip.transpose_(0, 1)  # [num_channels, seq_len]
            data_samples.append({"imu": clip, "label": self.label2id[scenario]})

        return data_samples

    def _process_line(self, line):
        if self.dataset_name == "UESTC-MMEA-CL":
            return self._process_line_uestc(line)
        elif self.dataset_name == "ego4d_data":
            return self._process_line_ego4d(line)
        elif self.dataset_name == "egoexo4d":
            return self._process_line_ego4d(line)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")


class VideoIMUDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_name,
        videos_path,
        imu_path,
        label2id,
        clip_sampler,
        video_transforms: Compose,
        dataset_name: str,
        imu_is_raw: bool = True,
        num_clips: int = 0,
    ):
        self.videos_path = videos_path
        self.imu_path = imu_path
        self.imu_is_raw = imu_is_raw
        self.label2id = label2id
        self.Racc = 16384
        self.Rgyro = 16.4
        self.video_path_handler = VideoPathHandler()
        self.video_transforms = video_transforms
        self.clip_sampler = clip_sampler
        self.dataset_name = dataset_name
        self.num_clips = num_clips

        with open(file_name, "r") as f:
            lines = f.readlines()

        self.data = []
        for line in lines:
            self.data.extend(self._process_line(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.dataset_name == "ego4d_data" or self.dataset_name == "egoexo4d":
            item = self.data[idx]
            video_path = item['video_path']
            imu_path = item['imu_path']
            label = item['label']
            clip_start = item['clip_start']
            clip_end = item['clip_end']
            imu_start = item['imu_start']
            imu_end = item['imu_end']

            imu_data = self._load_imu_clip(imu_path, imu_start, imu_end)

            video_data = self._load_video_clip(video_path, clip_start, clip_end)

            return {
                "video": video_data,
                "imu": imu_data,
                "label": label,
            }

        elif self.dataset_name == "UESTC-MMEA-CL":
            item = self.data[idx]
            video_path = item['video_path']
            imu_path = item['imu_path']
            label = item["label"]

            df = pd.read_csv(imu_path)
            if self.imu_is_raw:
                accel_data = df.iloc[:, :3] / self.Racc
                gyro_data = df.iloc[:, 3:6] / self.Rgyro
                imu_data = pd.concat([accel_data, gyro_data], axis=1).values
            else:
                imu_data = df.iloc[:, :6].values

            imu_data = np.transpose(imu_data, (1, 0)) # [num_channels, seq_len]

            seq_len = imu_data.shape[0]

            video = self.video_path_handler.video_from_path(
                video_path, decode_audio=False, decoder="pyav"
            )
            loaded_clip = video.get_clip(0.00, video.duration)
            video_data = self.video_transforms(loaded_clip)["video"]

            return {
                "video": video_data,
                "imu": imu_data,
                "seq_len": seq_len,
                "label": label,
            }

    def _process_line_uestc(self, line):
        line = line.strip()
        original_path, *_ = line.split()
        original_path = original_path.replace("yourdataset_path/data/", "")
        video_path = (self.videos_path / original_path).with_suffix(".mp4")
        imu_path = (self.imu_path / original_path).with_suffix(".csv")
        class_label = video_path.parent.name.split("_", 1)[1]
        label_id = self.label2id[class_label]

        return [
            {
                "video_path": video_path,
                "imu_path": imu_path,
                "label": label_id,
            }
        ]

    def _process_line_ego4d(self, line):
        uid, scenario = line.strip().split("\t")
        scenario = scenario.strip()
        video_path = self.videos_path / f"{uid}.mp4"
        imu_path = self.imu_path / f"{uid}.npy"

        # [num_classes, seq_len] -> [seq_len, num_classes]
        imu_data = np.load(imu_path).T
        imu_data = torch.tensor(imu_data, dtype=torch.float32)

        steps_per_clip = int(5 * 200)
        num_clips = imu_data.shape[0] // steps_per_clip

        video = self.video_path_handler.video_from_path(
            video_path, decode_audio=False, decoder="pyav"
        )
        video_duration = video.duration

        video_infos = []
        start_sec = 0.0
        while start_sec < video_duration:
            clip_info = self.clip_sampler(start_sec, video.duration, None)
            is_last_clip = clip_info.is_last_clip
            if is_last_clip:
                break
            start_sec = clip_info.clip_start_sec
            end_sec = clip_info.clip_end_sec
            video_infos.append((start_sec, end_sec))
            start_sec = end_sec

        num_clips = min(num_clips, len(video_infos))
        data_samples = []

        for i in range(num_clips):
            imu_start = i * steps_per_clip
            imu_end = imu_start + steps_per_clip

            clip_start, clip_end = video_infos[i]

            data_samples.append({
                "video_path": video_path,
                "imu_path": imu_path,
                "label": self.label2id[scenario],
                "clip_start": clip_start,
                "clip_end": clip_end,
                "imu_start": imu_start,
                "imu_end": imu_end,
            })


        if self.num_clips > 0 and len(data_samples) > self.num_clips:
            np.random.RandomState(16349).shuffle(data_samples)
            data_samples = data_samples[: self.num_clips]

        return data_samples

    def _process_line(self, line):
        if self.dataset_name == "UESTC-MMEA-CL":
            return self._process_line_uestc(line)
        elif self.dataset_name == "ego4d_data":
            return self._process_line_ego4d(line)
        elif self.dataset_name == "egoexo4d":
            return self._process_line_ego4d(line)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
    def _load_imu_clip(self, imu_path, start_idx, end_idx):
        imu_data = np.load(imu_path)[:, start_idx:end_idx]
        imu_data = torch.tensor(imu_data, dtype=torch.float32)
        return imu_data

    def _load_video_clip(self, video_path, clip_start, clip_end):
        video = self.video_path_handler.video_from_path(
            video_path, decode_audio=False, decoder="pyav"
        )
        loaded_clip = video.get_clip(clip_start, clip_end)
        video_data = self.video_transforms(loaded_clip)["video"]
        return video_data


def get_imu_label_train_dataloader(
    batch_size, imu_path, label2id, train_file_name, dataset, imu_is_raw, imu_ckpt_name=None
):
    if "uestc" not in dataset.lower():
        imu_label_collate_fn = eval_collate_fn
    elif "mantis" in imu_ckpt_name.lower():
        imu_label_collate_fn = UESTC_MMEA_CL_Mantis_eval_collate_fn
    elif "moment" in imu_ckpt_name.lower():
        imu_label_collate_fn = UESTC_MMEA_CL_MOMENT_eval_collate_fn

    imu_label_train_dataset = IMUDataset(
        train_file_name,
        imu_path,
        label2id,
        dataset,
        imu_is_raw,
    )

    imu_label_train_dataloader = torch.utils.data.DataLoader(
        imu_label_train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=imu_label_collate_fn,
        # num_workers=2,
    )
    return imu_label_train_dataloader


def get_imu_label_test_dataloader(
    batch_size, imu_path, label2id, test_file_name, dataset, imu_is_raw, imu_ckpt_name=None
):
    if "uestc" not in dataset.lower():
        imu_label_collate_fn = eval_collate_fn
    elif "mantis" in imu_ckpt_name.lower():
        imu_label_collate_fn = UESTC_MMEA_CL_Mantis_eval_collate_fn
    elif "moment" in imu_ckpt_name.lower():
        imu_label_collate_fn = UESTC_MMEA_CL_MOMENT_eval_collate_fn

    imu_label_test_dataset = IMUDataset(
        test_file_name,
        imu_path,
        label2id,
        dataset,
        imu_is_raw,
    )

    imu_label_test_dataloader = torch.utils.data.DataLoader(
        imu_label_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=imu_label_collate_fn,
        # num_workers=2,
    )
    return imu_label_test_dataloader


def video_collate_fn(examples):
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    return {"video": pixel_values}

def get_video_imu_train_dataloader(
    encode_batch_size,
    train_file_name,
    videos_path,
    imu_path,
    label2id,
    clip_sampler,
    train_transform,
    is_raw,
    dataset_name,
    imu_ckpt_name=None,
    num_clips=None,
):  
    if "uestc" not in dataset_name.lower():
        encode_collate_fn = EGO_encode_train_collate_fn
        print("Using EGO collate function")
    elif "mantis" in imu_ckpt_name.lower():
        encode_collate_fn = UESTC_MMEA_CL_Mantis_encode_train_collate_fn
        print("Using Mantis collate function")
    elif "moment" in imu_ckpt_name.lower():
        encode_collate_fn = UESTC_MMEA_CL_MOMENT_encode_train_collate_fn
        print("Using MOMENT collate function")
    video_imu_train_dataset = VideoIMUDataset(
        train_file_name,
        videos_path,
        imu_path,
        label2id,
        clip_sampler=clip_sampler,
        video_transforms=train_transform,
        imu_is_raw=is_raw,
        dataset_name=dataset_name,
        num_clips=num_clips,
    )
    video_imu_train_dataloader = torch.utils.data.DataLoader(
        video_imu_train_dataset,
        batch_size=encode_batch_size,
        shuffle=False,
        collate_fn=encode_collate_fn,
        num_workers=2,
    )

    return video_imu_train_dataloader
