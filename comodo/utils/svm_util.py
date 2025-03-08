import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from tqdm import tqdm

def extract_embeddings(dataloader, model, device):
    embeddings, labels = [], []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            # [batch_size, num_channels, seq_len]
            imu_data = batch["imu"].to(device)
            batch_labels = batch["label"]

            # [batch_size, hidden_size]
            batch_embeddings = model(imu_data)

            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

            embeddings.extend(batch_embeddings.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())

    return np.array(embeddings), np.array(labels)

def train_svm_on_embeddings(embeddings, labels, max_samples=10000):
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    svm = SVC(C=100000, gamma="scale", probability=True)

    if embeddings.shape[0] // np.unique(labels).shape[0] < 5 or embeddings.shape[0] < 50:
        svm.fit(embeddings, labels)
    else:
        param_grid = {
            "C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
            "kernel": ["rbf"],
            "degree": [3],
            "gamma": ["scale"],
            "coef0": [0],
            "shrinking": [True],
            "probability": [True],
            "tol": [0.001],
            "cache_size": [200],
            "class_weight": [None],
            "verbose": [False],
            "max_iter": [10000000],
            "decision_function_shape": ["ovr"],
        }

        if embeddings.shape[0] > max_samples:
            embeddings, _, labels, _ = train_test_split(
                embeddings, labels, train_size=max_samples, random_state=0, stratify=labels
            )

        grid_search = GridSearchCV(
            svm, param_grid, cv=5, n_jobs=10
        )
        grid_search.fit(embeddings, labels)
        svm = grid_search.best_estimator_

    return svm, scaler

def evaluate_svm(svm, scaler, embeddings, labels):
    embeddings = scaler.transform(embeddings)

    # Get probabilities for all classes
    probabilities = svm.predict_proba(embeddings)
    # Calculate top-1 accuracy (acc@1)
    predictions = np.argmax(probabilities, axis=1)
    acc1 = accuracy_score(labels, predictions)
    print(f"SVM Top-1 Accuracy (acc@1): {acc1}")

    acc3 = top_k_accuracy_score(labels, probabilities, k=3)
    print(f"SVM Top-3 Accuracy (acc@3): {acc3}")

    # Calculate top-5 accuracy (acc@5)
    acc5 = top_k_accuracy_score(labels, probabilities, k=5)
    print(f"SVM Top-5 Accuracy (acc@5): {acc5}")

    return acc1, acc3, acc5