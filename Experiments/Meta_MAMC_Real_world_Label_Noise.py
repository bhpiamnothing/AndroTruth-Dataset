

import re
import csv
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset



# INPUT OUR ANDROTRUTH FEATURE CSV PATH HERE
FEATURES_CSV = r"Experiments\AndroTruth_feature_3000_matrix_df.csv"
# INPUT KASPERSKY NOISY LABEL CSV PATH HERE, you can change to other noisy label sources if needed (e.g., avlcass2,claravy etc.) For GT labels experiment, please use AndroTruth_labels.csv
NOISY_LABELS_CSV = r"Experiments\AndroTruth_kaspersky_datasets.csv"
# INPUT OUR ANDROTRUTH CLEAN LABEL CSV PATH HERE
CLEAN_LABELS_CSV = r"Experiments\AndroTruth_labels.csv"

SHA_COL = "sha256"
FAMILY_COL = "family"

SEED = 42
DEVICE = "auto"
N_FOLDS = 5


N_WAY = 5
K_SHOT = 5
Q_QUERY = 10
P_MIX = 0.25

INNER_STEPS = 1
INNER_LR = 0.01
META_LR = 1e-3
META_EPOCHS = 30
EPS_STOP = 0.01
STOP_PATIENCE = 3

FT_EPOCHS = 15
FT_LR = 1e-3
BATCH_SIZE = 256


SAVE_PER_FAMILY_CSV = ""
SAVE_CONFUSION_CSV = ""


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pick_device(arg: str = "auto"):
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_family(x: str) -> str:
    """Fix case/whitespace variants so same family isn't split into multiple classes."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.casefold()
    return s



def load_feature_matrix_csv(feature_csv: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(feature_csv)
    if SHA_COL not in df.columns:
        raise RuntimeError(f"{feature_csv} must contain column '{SHA_COL}'")
    sha_list = df[SHA_COL].astype(str).values
    feat_cols = [c for c in df.columns if c != SHA_COL]
    if not feat_cols:
        raise RuntimeError("No feature columns found (CSV only has sha256?)")
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32).values
    return sha_list, X

def load_label_map(labels_csv: str) -> Dict[str, str]:
    df = pd.read_csv(labels_csv)
    if SHA_COL not in df.columns or FAMILY_COL not in df.columns:
        raise RuntimeError(f"{labels_csv} must contain columns '{SHA_COL}' and '{FAMILY_COL}'")
    df = df[[SHA_COL, FAMILY_COL]].copy()
    df[SHA_COL] = df[SHA_COL].astype(str)
    df[FAMILY_COL] = df[FAMILY_COL].map(normalize_family)
    return dict(zip(df[SHA_COL].values, df[FAMILY_COL].values))

def align_pool(sha_list: np.ndarray, X: np.ndarray,
               noisy_map: Dict[str, str], clean_map: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.array([(s in noisy_map) and (s in clean_map) for s in sha_list], dtype=bool)
    if mask.sum() == 0:
        raise RuntimeError("No samples have BOTH noisy and clean labels after sha256 alignment.")
    X2 = X[mask]
    sha2 = sha_list[mask]
    y_noisy_str = np.array([noisy_map[s] for s in sha2], dtype=object)
    y_clean_str = np.array([clean_map[s] for s in sha2], dtype=object)
    return X2, y_noisy_str, y_clean_str



def functional_forward(params: List[torch.Tensor], x: torch.Tensor) -> torch.Tensor:

    x = F.linear(x, params[0], params[1])
    x = F.relu(x)
    x = F.linear(x, params[2], params[3])
    x = F.relu(x)
    x = F.linear(x, params[4], params[5])
    return x

def inner_update_params(params: List[torch.Tensor], x: torch.Tensor, y: torch.Tensor, lr: float, steps: int) -> List[torch.Tensor]:
    for _ in range(steps):
        pred = functional_forward(params, x)
        loss = F.cross_entropy(pred, y)
        grad = torch.autograd.grad(loss, params, create_graph=True)
        params = [p - lr * g for p, g in zip(params, grad)]
    return params



def application_based_sample(class_to_idx: Dict[int, np.ndarray],
                             n_way: int, k_shot: int, q_query: int,
                             X: np.ndarray, y: np.ndarray):
    all_indices = np.arange(len(y))
    np.random.shuffle(all_indices)
    selected_indices = all_indices[: n_way * (k_shot + q_query)]

    selected_y = y[selected_indices]
    unique_classes = np.unique(selected_y)
    if len(unique_classes) < n_way:
        additional = np.setdiff1d(np.unique(y), unique_classes)
        np.random.shuffle(additional)
        additional = additional[: (n_way - len(unique_classes))]
        for cls in additional:
            if cls in class_to_idx and len(class_to_idx[cls]) > 0:
                cls_idx = np.random.choice(class_to_idx[cls], 1)[0]
                selected_indices = np.append(selected_indices, cls_idx)

    support_idx, support_y = [], []
    query_idx, query_y = [], []

    buckets = {}
    for i in selected_indices:
        buckets.setdefault(y[i], []).append(i)

    selected_classes = list(buckets.keys())[:n_way]
    for cls in selected_classes:
        idxs = buckets[cls]
        np.random.shuffle(idxs)
        s_num = min(k_shot, len(idxs) // 2)
        q_num = min(q_query, len(idxs) - s_num)
        if s_num <= 0 or q_num <= 0:
            continue
        support_idx.extend(idxs[:s_num])
        support_y.extend([cls] * s_num)
        query_idx.extend(idxs[s_num:s_num + q_num])
        query_y.extend([cls] * q_num)

    if len(support_idx) == 0 or len(query_idx) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    return X[np.array(support_idx)], np.array(support_y), X[np.array(query_idx)], np.array(query_y)

def family_based_sample(class_to_idx: Dict[int, np.ndarray],
                        n_way: int, k_shot: int, q_query: int,
                        X: np.ndarray):
    candidates = [c for c in class_to_idx if len(class_to_idx[c]) >= (k_shot + q_query)]
    if len(candidates) < n_way:
        candidates = [c for c in class_to_idx if len(class_to_idx[c]) >= (k_shot + 1)]
        if not candidates:
            return np.array([]), np.array([]), np.array([]), np.array([])

    n_way_eff = min(n_way, len(candidates))
    selected_classes = np.random.choice(candidates, n_way_eff, replace=False)

    support_idx, support_y = [], []
    query_idx, query_y = [], []

    for cls in selected_classes:
        idxs = class_to_idx[cls].copy()
        np.random.shuffle(idxs)
        actual_k = min(k_shot, len(idxs) - 1)
        actual_q = min(q_query, len(idxs) - actual_k)
        if actual_k <= 0 or actual_q <= 0:
            continue
        s_idx = idxs[:actual_k]
        q_idx = idxs[actual_k: actual_k + actual_q]
        support_idx.extend(s_idx.tolist())
        support_y.extend([cls] * len(s_idx))
        query_idx.extend(q_idx.tolist())
        query_y.extend([cls] * len(q_idx))

    if len(support_idx) == 0 or len(query_idx) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    return X[np.array(support_idx)], np.array(support_y), X[np.array(query_idx)], np.array(query_y)

def sample_task(class_to_idx: Dict[int, np.ndarray],
                n_way: int, k_shot: int, q_query: int, p: float,
                X: np.ndarray, y: np.ndarray):
    if np.random.rand() < p:
        return application_based_sample(class_to_idx, n_way, k_shot, q_query, X, y)
    return family_based_sample(class_to_idx, n_way, k_shot, q_query, X)



@dataclass
class PerFamilyMetrics:
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0
    total_tn: int = 0
    acc: float = 0.0
    prec: float = 0.0
    rec: float = 0.0
    f1: float = 0.0



def main():
    set_seed(SEED)
    device = pick_device(DEVICE)
    print(f"[*] Device: {device}")


    sha_list, X = load_feature_matrix_csv(FEATURE_CSV)
    noisy_map = load_label_map(NOISY_LABELS_CSV)
    clean_map = load_label_map(CLEAN_LABELS_CSV)

    X, y_noisy_str, y_clean_str = align_pool(sha_list, X, noisy_map, clean_map)
    print(f"[*] Aligned pool: samples={X.shape[0]}, features={X.shape[1]}")
    print(f"[*] Noisy==Clean agreement (normalized): {np.mean(y_noisy_str == y_clean_str):.2%}")


    le = LabelEncoder()
    le.fit(np.concatenate([y_noisy_str, y_clean_str], axis=0))
    y_noisy = le.transform(y_noisy_str)
    y_clean = le.transform(y_clean_str)

    classes = le.classes_
    num_classes = len(classes)
    input_dim = X.shape[1]
    print(f"[*] Classes: {num_classes}")


    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    all_acc, all_f1, all_rec = [], [], []
    all_confusions = []
    all_per_family: Dict[str, List[PerFamilyMetrics]] = {c: [] for c in classes}

    for fold_id, (train_idx, test_idx) in enumerate(skf.split(np.arange(len(y_clean)), y_clean), start=1):
        fold_seed = SEED + fold_id * 100
        set_seed(fold_seed)

        print(f"\n===== Fold {fold_id}/{N_FOLDS} (seed={fold_seed}) =====")


        X_train = X[train_idx]
        y_train = y_noisy[train_idx]
        X_test = X[test_idx]
        y_test = y_clean[test_idx]

        train_class_to_idx = {c: np.flatnonzero(y_train == c) for c in np.unique(y_train)}


        model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=META_LR)

        prev_loss = float("inf")
        patience_cnt = 0

        for epoch in range(1, META_EPOCHS + 1):
            model.train()
            meta_loss = 0.0
            valid_tasks = 0

            num_tasks = max(1, len(y_train) // max(1, (N_WAY * (K_SHOT + Q_QUERY))))

            for _ in range(num_tasks):
                support_x, support_y, query_x, query_y = sample_task(
                    train_class_to_idx, N_WAY, K_SHOT, Q_QUERY, P_MIX, X_train, y_train
                )
                if support_x.size == 0 or query_x.size == 0:
                    continue

                support_x_t = torch.from_numpy(support_x).to(device)
                support_y_t = torch.from_numpy(support_y).long().to(device)
                query_x_t = torch.from_numpy(query_x).to(device)
                query_y_t = torch.from_numpy(query_y).long().to(device)

                base_params = [p.clone().detach().requires_grad_(True) for p in model.parameters()]
                fast_params = inner_update_params(base_params, support_x_t, support_y_t, INNER_LR, INNER_STEPS)

                query_pred = functional_forward(fast_params, query_x_t)
                task_loss = F.cross_entropy(query_pred, query_y_t)
                if torch.isnan(task_loss) or torch.isinf(task_loss):
                    continue

                meta_loss += task_loss
                valid_tasks += 1

            if valid_tasks > 0:
                meta_loss = meta_loss / valid_tasks
            else:
                meta_loss = torch.tensor(0.0, device=device, requires_grad=True)

            optimizer.zero_grad()
            meta_loss.backward()
            optimizer.step()

            print(f"[*] Meta Epoch {epoch}/{META_EPOCHS}: Loss {meta_loss.item():.4f}")

            if abs(prev_loss - meta_loss.item()) < EPS_STOP:
                patience_cnt += 1
                if patience_cnt >= STOP_PATIENCE:
                    print(f"[*] Early stop at epoch {epoch}")
                    break
            else:
                patience_cnt = 0
            prev_loss = meta_loss.item()


        model.train()
        ft_opt = torch.optim.Adam(model.parameters(), lr=FT_LR)
        ft_dataset = TensorDataset(
            torch.from_numpy(X_train.astype(np.float32)),
            torch.from_numpy(y_train).long()
        )
        ft_loader = DataLoader(ft_dataset, batch_size=BATCH_SIZE, shuffle=True)

        for ft_epoch in range(1, FT_EPOCHS + 1):
            ft_loss = 0.0
            for bx, by in ft_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                loss = F.cross_entropy(pred, by)
                ft_opt.zero_grad()
                loss.backward()
                ft_opt.step()
                ft_loss += loss.item()
            print(f"[*] FT Epoch {ft_epoch}/{FT_EPOCHS}: Loss {ft_loss / max(1, len(ft_loader)):.4f}")


        model.eval()
        with torch.no_grad():
            test_x = torch.from_numpy(X_test.astype(np.float32)).to(device)
            test_pred = model(test_x).argmax(dim=1).cpu().numpy()

        acc = accuracy_score(y_test, test_pred)
        f1m = f1_score(y_test, test_pred, average="macro", zero_division=0)
        recm = recall_score(y_test, test_pred, average="macro", zero_division=0)
        print(f"[*] Test(clean): Acc {acc:.4f}, Macro-F1 {f1m:.4f}, Macro-Recall {recm:.4f}")

        all_acc.append(acc); all_f1.append(f1m); all_rec.append(recm)


        conf = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_test, test_pred):
            conf[t, p] += 1
        all_confusions.append(conf)


        for c_idx, c_name in enumerate(classes):
            tp = conf[c_idx, c_idx]
            fp = conf[:, c_idx].sum() - tp
            fn = conf[c_idx, :].sum() - tp
            tn = conf.sum() - tp - fp - fn

            prec = tp / (tp + fp) if tp + fp > 0 else 0.0
            rec = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1c = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
            accc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

            all_per_family[c_name].append(PerFamilyMetrics(
                total_tp=tp, total_fp=fp, total_fn=fn, total_tn=tn,
                acc=accc, prec=prec, rec=rec, f1=f1c
            ))


    print("\n===== 5-Fold CV Summary (train=noisy, test=clean) =====")
    print(f"Acc mean±std: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"Macro-F1 mean±std: {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
    print(f"Macro-Recall mean±std: {np.mean(all_rec):.4f} ± {np.std(all_rec):.4f}")

    if SAVE_PER_FAMILY_CSV:
        with open(SAVE_PER_FAMILY_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Family", "Avg_Acc", "Avg_Prec", "Avg_Rec", "Avg_F1"])
            for fam, ms in sorted(all_per_family.items(), key=lambda x: len(x[1]), reverse=True):
                w.writerow([
                    fam,
                    f"{np.mean([m.acc for m in ms]):.4f}",
                    f"{np.mean([m.prec for m in ms]):.4f}",
                    f"{np.mean([m.rec for m in ms]):.4f}",
                    f"{np.mean([m.f1 for m in ms]):.4f}",
                ])
        print(f"[*] Saved per-family metrics: {SAVE_PER_FAMILY_CSV}")

    if SAVE_CONFUSION_CSV:
        avg_conf = np.mean(all_confusions, axis=0)
        df_conf = pd.DataFrame(avg_conf, index=classes, columns=classes)
        df_conf.to_csv(SAVE_CONFUSION_CSV)
        print(f"[*] Saved confusion matrix: {SAVE_CONFUSION_CSV}")


if __name__ == "__main__":
    main()
