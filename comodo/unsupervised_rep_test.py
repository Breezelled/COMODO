from utils.dataloader_util import get_imu_label_test_dataloader, get_imu_label_train_dataloader

from utils.model_util import imu_student_mlp, imu_student, IMUStudent, IMUStudentMLP

from utils.svm_util import (
    extract_embeddings,
    train_svm_on_embeddings,
    evaluate_svm,
)

import torch
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
import json
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = ArgumentParser()
args.add_argument("--dataset_path", type=str, default="dataset/egoexo4d")
args.add_argument("--imu_ckpt", type=str, default="AutonLab/MOMENT-1-small")
args.add_argument("--model_path", type=str, default="")

def append_to_json(new_entry, category, json_file="results.json"):
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}
    else:
        existing_data = {}

    if category not in existing_data:
        existing_data[category] = []

    existing_data[category].append(new_entry)

    with open(json_file, "w") as f:
        json.dump(existing_data, f, indent=4)

    print(f"Results appended to {json_file}")

def svm_test(train_dataloader, test_dataloader, imu_model, device, model_path, dataset, imu_ckpt):
    imu_model.remove_projection_layer()

    print("Extracting embeddings for SVM training...")
    train_embeddings, train_labels = extract_embeddings(
        train_dataloader, imu_model, device
    )

    print("Training SVM on extracted embeddings...")
    svm, scaler = train_svm_on_embeddings(train_embeddings, train_labels)

    print("Evaluating SVM on test set...")
    test_embeddings, test_labels = extract_embeddings(
        test_dataloader, imu_model, device
    )

    acc1, acc3, acc5 = evaluate_svm(svm, scaler, test_embeddings, test_labels)

    results = {
        "imu": imu_ckpt,
        "dataset": dataset,
        "model": model_path,
        "acc1": acc1,
        "acc3": acc3,
        "acc5": acc5
    }

    if isinstance(imu_model, IMUStudent):
        category = "IMU2CLIP&L2"
    elif isinstance(imu_model, IMUStudentMLP):
        category = "COMODO"
    else:
        category = "OTHER"

    append_to_json(results, category)


def unsupervised_rep_test(batch_size, imu_path, label2id, train_file_name, test_file_name, dataset, imu_is_raw, imu_ckpt, model_path):
    if "infonce" in model_path:
        imu_model = imu_student(768, imu_ckpt, 8, None, 6)
    else:
        imu_model = imu_student_mlp(128, imu_ckpt, 8, 2048, model_path)

    if model_path != "" and "full" not in model_path: # mantis supervised finetuned wont load from here
        imu_model.load_state_dict(torch.load(model_path, map_location=device))

    train_dataloader = get_imu_label_train_dataloader(batch_size, imu_path, label2id, train_file_name, dataset, imu_is_raw, imu_ckpt)
    test_dataloader = get_imu_label_test_dataloader(batch_size, imu_path, label2id, test_file_name, dataset, imu_is_raw, imu_ckpt)
    
    svm_test(train_dataloader, test_dataloader, imu_model, device, model_path, dataset, imu_ckpt)

def main(args):
    dataset_path = args.dataset_path
    path = Path(dataset_path)
    dataset = args.dataset_path.split("/")[-1]
    if dataset == "UESTC-MMEA-CL":
        imu_path = path / "sensor"
        class_labels = sorted(p.name for p in imu_path.glob("*"))
        label2id = {
            label.split("_", 1)[1]: int(label.split("_", 1)[0]) - 1
            for label in class_labels
        }
        is_raw = True
    elif dataset == "ego4d_data":
        imu_path = path / "v2/processed_imu"
        class_labels = sorted(
            pd.read_csv(path / "train.txt", sep="\t", header=None)[1].unique()
        )
        label2id = {label: idx for idx, label in enumerate(class_labels)}
        is_raw = False
    elif dataset == "egoexo4d":
        imu_path = path / "processed_imu"
        class_labels = sorted(
            pd.read_csv(path / "train.txt", sep="\t", header=None)[1].unique()
        )
        label2id = {label: idx for idx, label in enumerate(class_labels)}
        is_raw = False

    print("Dataset: ", dataset)
    print("IMU", args.imu_ckpt)
    print("Model", args.model_path)
    unsupervised_rep_test(48, imu_path, label2id, path / "train.txt", path / "test.txt", dataset, is_raw, args.imu_ckpt, args.model_path)

if __name__ == "__main__":
    main(args.parse_args())