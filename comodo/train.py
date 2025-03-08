from argparse import ArgumentParser
import numpy as np
import random
from joblib import load
import pickle
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from transformers import (
    get_linear_schedule_with_warmup,
)

from loss import COMODOLoss

from mantis.trainer import MantisTrainer

from utils.model_util import VideoTeacherMLP
from utils.model_util import IMUStudentMLP
from utils.model_util import create_pipeline

from unsupervised_rep_test import svm_test
from utils.dataloader_util import (
    get_imu_label_train_dataloader,
    get_imu_label_test_dataloader,
    get_video_imu_train_dataloader,
    VideoIMUDataset,
)
from utils.clipsampler_util import (
    get_uestc_video_transform_clipsampler,
    get_ego4d_video_transform_clipsampler,
)

from utils.collate_util import *

args = ArgumentParser()
args.add_argument("--video_ckpt", type=str)
args.add_argument("--imu_ckpt", type=str)
args.add_argument("--dataset_path", type=str, default="dataset/egoexo4d")
args.add_argument("--is_raw", action="store_true", default=True)
args.add_argument("--queue_size", type=int, default=16384)
args.add_argument("--student_temp", type=float, default=5)
args.add_argument("--teacher_temp", type=float, default=5)
args.add_argument("--learning_rate", type=float, default=3e-4)
args.add_argument("--num_epochs", type=int, default=20)
args.add_argument("--batch_size", type=int, default=32)
args.add_argument(
    "--num_clips",
    type=int,
    default=0,
    help="Number of clips to randomly keep in each sample. If set to 0, all clips will be kept",
)
args.add_argument("--seed", type=int, default=42)
args.add_argument("--encoded_video_path", type=str, default="data/encoded.pkl")
args.add_argument("--anchor_video_path", type=str, default="data/anchor.pkl")
args.add_argument("--mlp_hidden_dim", type=int, default=2048)
args.add_argument("--mlp_output_dim", type=int, default=128)
args.add_argument("--reduction", type=str, default="concat")


def evaluate(imu_student, val_dataloader, video_embedding):
    device = next(imu_student.parameters()).device
    imu_student.eval()
    if isinstance(imu_student, MantisTrainer):
        imu_student.network.eval()
    total, correct = 0, 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Evaluating"):
            imu_data = batch["imu"].to(device)
            label = batch["label"].to(device)
            input_mask = batch.get("input_mask")
            input_mask = input_mask.to(device) if input_mask is not None else None

            # [batch_size, hidden_size]
            imu_embedding = imu_student(imu_data, input_mask)
            # normalize embeddings
            imu_embedding = F.normalize(imu_embedding, p=2, dim=1)

            # [batch_size, hidden_size] x [hidden_size, num_classes] -> [batch_size, num_classes]
            cosine_similarity = torch.matmul(imu_embedding, video_embedding.T)
            # [batch_size]
            predicted_indices = torch.argmax(cosine_similarity, dim=1).cpu().numpy()

            correct += sum(p == l for p, l in zip(predicted_indices, label))
            total += len(label)

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy}")


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    print("Starting main function")
    set_random_seed(args.seed)
    imu_ckpt_name = args.imu_ckpt.split("/")[1]
    video_ckpt_name = args.video_ckpt.split("/")[1]
    dataset = args.dataset_path.split("/")[-1]
    if args.video_ckpt.split("/")[0] == "model":
        args.encoded_video_path += f"_{video_ckpt_name}_{dataset}_finetuned.pkl"
        args.anchor_video_path += f"_{video_ckpt_name}_{dataset}_finetuned.pkl"
    else:
        args.encoded_video_path += f"_{video_ckpt_name}_{dataset}_pretrained.pkl"
        args.anchor_video_path += f"_{video_ckpt_name}_{dataset}_pretrained.pkl"
    print(args)
    n_channels = 6
    # data args
    dataset_path = args.dataset_path
    encoded_video_path = args.encoded_video_path

    if "videomae" in args.video_ckpt.lower():
        sample_rate = 3.125
    elif "timesformer" in args.video_ckpt.lower():
        sample_rate = 6.25
    fps = 25 if "UESTC" in dataset else 10
    encode_batch_size = args.batch_size

    path = Path(dataset_path)
    train_file_name = path / "train.txt"
    val_file_name = path / "val.txt"
    test_file_name = path / "test.txt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.init()

    print("Loading teacher model")
    video_teacher = VideoTeacherMLP(
        args.video_ckpt, args.mlp_output_dim, args.mlp_hidden_dim, device=device
    )

    def get_dataset_config(dataset_name, base_path):
        if dataset_name == "ego4d_data":
            videos_path = base_path / "v2/processed_video"
            imus_path = base_path / "v2/processed_imu"
            class_labels = sorted(
                pd.read_csv(base_path / "train.txt", sep="\t", header=None)[1].unique()
            )
            label2id = {label: idx for idx, label in enumerate(class_labels)}
            id2label = {v: k for k, v in label2id.items()}
            train_transform, val_transform, clip_sampler, clip_sampler = (
                get_ego4d_video_transform_clipsampler(video_teacher, sample_rate, fps)
            )
            return (
                videos_path,
                imus_path,
                label2id,
                id2label,
                train_transform,
                val_transform,
                clip_sampler,
                clip_sampler,
            )
        elif dataset_name == "UESTC-MMEA-CL":
            videos_path = base_path / "video"
            imus_path = base_path / "sensor"
            class_labels = sorted(p.name for p in videos_path.glob("*"))
            label2id = {
                label.split("_", 1)[1]: int(label.split("_", 1)[0]) - 1
                for label in class_labels
            }
            id2label = {v: k for k, v in label2id.items()}
            train_transform, val_transform, train_sampler, val_sampler = (
                get_uestc_video_transform_clipsampler(video_teacher, sample_rate, fps)
            )

            return (
                videos_path,
                imus_path,
                label2id,
                id2label,
                train_transform,
                val_transform,
                train_sampler,
                val_sampler,
            )
        elif dataset_name == "egoexo4d":
            videos_path = base_path / "processed_video"
            imus_path = base_path / "processed_imu"
            class_labels = sorted(
                pd.read_csv(base_path / "train.txt", sep="\t", header=None)[1].unique()
            )
            label2id = {label: idx for idx, label in enumerate(class_labels)}
            id2label = {v: k for k, v in label2id.items()}
            train_transform, val_transform, clip_sampler, clip_sampler = (
                get_ego4d_video_transform_clipsampler(video_teacher, sample_rate, fps)
            )
            return (
                videos_path,
                imus_path,
                label2id,
                id2label,
                train_transform,
                val_transform,
                clip_sampler,
                clip_sampler,
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    print("Getting dataset config")
    (
        videos_path,
        imu_path,
        label2id,
        id2label,
        trian_transform,
        val_transform,
        train_sampler,
        val_sampler,
    ) = get_dataset_config(dataset, path)
    num_classes = len(id2label)

    print("Loading student model")
    imu_student = create_pipeline(
        args.imu_ckpt,
        num_classes,
        device,
        args.reduction,
        n_channels,
    )

    imu_student = IMUStudentMLP(
        imu_student,
        device,
        args.mlp_output_dim,
        args.mlp_hidden_dim,
        activation_fn=nn.GELU,
        reduction=args.reduction,
    )

    imu_student.to(device)

    video_imu_val_dataset = VideoIMUDataset(
        val_file_name,
        videos_path,
        imu_path,
        label2id,
        clip_sampler=val_sampler,
        video_transforms=val_transform,
        imu_is_raw=args.is_raw,
        dataset_name=dataset,
        num_clips=args.num_clips,
    )

    print("Initializing video imu dataloader")

    video_imu_train_dataloader = get_video_imu_train_dataloader(
        encode_batch_size,
        train_file_name,
        videos_path,
        imu_path,
        label2id,
        train_sampler,
        trian_transform,
        args.is_raw,
        dataset_name=dataset,
        imu_ckpt_name=imu_ckpt_name,
        num_clips=args.num_clips,
    )

    print(f"Video imu train dataset nums: {len(video_imu_train_dataloader.dataset)}")

    try:
        print("Loading encoded videos from file")
        encoded_videos = load(encoded_video_path)
        print("Loaded encoded videos from file")
    except:
        encoded_videos = []
        for batch in tqdm(video_imu_train_dataloader, desc="Encoding videos"):
            encoded_video = video_teacher.encode(batch["video"])
            encoded_videos.append(encoded_video.cpu().numpy())
        encoded_videos = np.concatenate(encoded_videos, axis=0)
        filename = encoded_video_path
        with open(filename, "wb") as f:
            pickle.dump(encoded_videos, f, protocol=4)

    print(f"Encoded videos num: {len(encoded_videos)}")

    video_indices = range(len(video_imu_train_dataloader.dataset))
    idxs_in_queue = set(
        np.random.RandomState(16349).choice(
            video_indices, args.queue_size, replace=False
        )
    )

    (
        train_samples,
        instance_queue_encoded,
        instance_queue_samples,
    ) = ([], [], [], [])

    for idx, encoded_video in enumerate(encoded_videos):
        imu = video_imu_train_dataloader.dataset[idx]["imu"]
        if idx not in idxs_in_queue:
            train_samples.append({"imu": imu, "encoded_video": encoded_video})
        else:
            instance_queue_encoded.append(encoded_video)
            instance_queue_samples.append({"imu": imu, "encoded_video": encoded_video})
    print(f"Train samples num: {len(train_samples)}")

    if "uestc" not in dataset.lower():
        t_collate_fn = train_collate_fn
        e_collate_fn = eval_collate_fn
    elif "mantis" in imu_ckpt_name.lower():
        t_collate_fn = UESTC_MMEA_CL_Mantis_train_collate_fn
        e_collate_fn = UESTC_MMEA_CL_Mantis_eval_collate_fn
    elif "moment" in imu_ckpt_name.lower():
        t_collate_fn = UESTC_MMEA_CL_MOMENT_train_collate_fn
        e_collate_fn = UESTC_MMEA_CL_MOMENT_eval_collate_fn

    train_dataloader = torch.utils.data.DataLoader(
        train_samples,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=t_collate_fn,
        # num_workers=2,
    )

    video_imu_val_dataloader = torch.utils.data.DataLoader(
        video_imu_val_dataset,
        batch_size=encode_batch_size,
        shuffle=False,
        collate_fn=e_collate_fn,
        # num_workers=2,
    )

    instance_queue_encoded = torch.tensor(np.stack(instance_queue_encoded, axis=0)).to(
        device
    )

    comodo_loss = COMODOLoss(
        instanceQ_encoded=instance_queue_encoded,
        student_model=imu_student,
        teacher_temp=args.teacher_temp,
        student_temp=args.student_temp,
    )

    del (
        encoded_videos,
        instance_queue_encoded,
        video_imu_train_dataloader,
    )

    print("Getting anchor video embeddings")
    # get anchor video embeddings per activity
    try:
        anchor_video_embeddings = load(args.anchor_video_path)
    except:
        anchor_video_embeddings = {label: None for label in range(num_classes)}
        for datum in tqdm(
            video_imu_val_dataset, desc="Getting anchor video embeddings"
        ):
            label = datum["label"]
            if anchor_video_embeddings[label] is None:
                video_tensor = datum["video"].permute(1, 0, 2, 3).unsqueeze(0)
                video_embedding = video_teacher.encode(video_tensor).squeeze(0)
                anchor_video_embeddings[label] = video_embedding
            else:
                continue

        with open(args.anchor_video_path, "wb") as f:
            pickle.dump(anchor_video_embeddings, f, protocol=4)

    del video_teacher

    num_training_steps = len(train_dataloader) + (
        len(train_dataloader) + len(instance_queue_samples)
    ) * (args.num_epochs - 1)

    num_warmup_steps = int(0.1 * num_training_steps)

    optimizer = torch.optim.AdamW(imu_student.parameters(), lr=args.learning_rate)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # [num_classes, hidden_size]
    video_embedding = torch.stack(
        [anchor_video_embeddings[label] for label in range(num_classes)]
    ).to(device)

    for epoch in range(args.num_epochs):
        total_loss = 0.0
        imu_student.train()

        # append initial queue sample to follow up training
        if epoch == 1:
            train_samples.extend(instance_queue_samples)
            train_dataloader = torch.utils.data.DataLoader(
                train_samples,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=t_collate_fn,
                # num_workers=2,
            )

        for idx, batch in enumerate(tqdm(train_dataloader)):
            imu = batch["imu"].to(device)
            input_mask = batch.get("input_mask")
            input_mask = input_mask.to(device) if input_mask is not None else None
            encoded_video = batch["encoded_video"].to(device)
            optimizer.zero_grad()
            loss = comodo_loss(imu, encoded_video, input_mask=input_mask)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {idx}, Loss {loss.item()}")

        print("Starting evaluation...")

        evaluate(imu_student, video_imu_val_dataloader, video_embedding)

    model_path = f"model/{dataset}_{args.reduction}_{args.queue_size}qs_{sample_rate}sr_{args.student_temp}stutmp_{args.teacher_temp}tchtmp_{args.learning_rate}lr_{args.num_epochs}epochs_{args.batch_size}bs_{args.mlp_hidden_dim}mhd_{args.mlp_output_dim}mod_{imu_ckpt_name}_{video_ckpt_name}.pth"
    print(args)
    print(model_path)
    torch.save(
        imu_student.state_dict(),
        model_path,
    )

    svm_test_dataloader = get_imu_label_test_dataloader(
        args.batch_size,
        imu_path,
        label2id,
        test_file_name,
        dataset,
        args.is_raw,
        imu_ckpt_name,
    )
    svm_train_dataloader = get_imu_label_train_dataloader(
        args.batch_size,
        imu_path,
        label2id,
        train_file_name,
        dataset,
        args.is_raw,
        imu_ckpt_name,
    )
    svm_test(
        svm_train_dataloader,
        svm_test_dataloader,
        imu_student,
        device,
        model_path=model_path,
        dataset=dataset,
        imu_ckpt=args.imu_ckpt,
    )


if __name__ == "__main__":
    main(args.parse_args())
