# -*- coding: utf-8 -*-
"""
Primary downstream evaluation for Meta-MAMC under multiple label sources:
  - GT
  - Kaspersky
  - AVClass2
  - ClarAVy

Primary protocol:
1. Build a joint pool aligned by sha256 between features, clean GT labels, and noisy sources.
2. Distinguish tool-unlabeled cases:
   - N/A -> missing_scan
   - null / benign -> no_malicious_detection
   - SINGLETON -> singleton
   - empty / [] / - -> missing
3. Globally exclude GT families with fewer than 5 samples BEFORE cross-validation.
4. Run standard 5-fold stratified CV on the retained GT families.
5. For noisy-label training, only source samples with explicit family labels are used.
6. Evaluation is always against filtered expert GT labels.

Metrics:
- ACC, Macro-Precision, Macro-Recall, Macro-F1
- AMI
- AVCLASS/Euphony-style Cluster-Precision / Recall / F1
"""

import os
import re
import random
import argparse
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    adjusted_mutual_info_score,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


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

PRIMARY_MIN_GT_FAMILY_SIZE = 5


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
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s.casefold()


def keyize(s: str) -> str:
    s = str(s).lower().strip()
    return re.sub(r"[^a-z0-9]+", "", s)


def normalize_for_display(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9._+-]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_synonyms(path):
    alias2canon_key = {}
    canon_key2display = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if not parts:
                continue
            canon = parts[0]
            canon_key = keyize(canon)
            canon_key2display[canon_key] = canon
            for p in parts:
                alias2canon_key[keyize(p)] = canon_key
    return alias2canon_key, canon_key2display


def make_canonizer(alias2canon_key, canon_key2display):
    def canon_key_func(lbl):
        if lbl is None or (isinstance(lbl, float) and pd.isna(lbl)):
            return None
        k = keyize(lbl)
        return alias2canon_key.get(k, k)

    def canonize(lbl):
        if lbl is None or (isinstance(lbl, float) and pd.isna(lbl)):
            return None
        ck = canon_key_func(lbl)
        return canon_key2display.get(ck, normalize_for_display(lbl))
    return canonize, canon_key_func


def normalize_tool_family(lbl):
    if lbl is None or (isinstance(lbl, float) and pd.isna(lbl)):
        return None, "missing"

    s = str(lbl).strip()
    s_lower = s.lower()

    if s == "" or s in {"-", "[]"}:
        return None, "missing"

    if s_lower == "n/a":
        return None, "missing_scan"

    if s_lower in {"null", "benign"}:
        return None, "no_malicious_detection"

    if s_lower == "singleton" or s_lower.startswith("singleton:"):
        return None, "singleton"

    return s, "labeled"


def load_feature_matrix_csv(feature_csv: str):
    df = pd.read_csv(feature_csv, low_memory=False)
    if SHA_COL not in df.columns:
        raise RuntimeError(f"{feature_csv} must contain column '{SHA_COL}'")
    df[SHA_COL] = df[SHA_COL].astype(str).str.lower()

    feat_cols = [c for c in df.columns if c != SHA_COL]
    if not feat_cols:
        raise RuntimeError("No feature columns found (CSV only has sha256?)")
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
    return df, feat_cols


def load_clean_labels(clean_csv: str, canon_key_func):
    df = pd.read_csv(clean_csv, low_memory=False)
    if SHA_COL not in df.columns or FAMILY_COL not in df.columns:
        raise RuntimeError(f"{clean_csv} must contain '{SHA_COL}' and '{FAMILY_COL}'")
    df = df[[SHA_COL, FAMILY_COL]].copy()
    df[SHA_COL] = df[SHA_COL].astype(str).str.lower()
    df["family_clean_raw"] = df[FAMILY_COL].astype(str)
    df["family_clean"] = df["family_clean_raw"].map(normalize_family)
    df["family_clean_ckey"] = df["family_clean"].map(canon_key_func)
    return df[[SHA_COL, "family_clean", "family_clean_ckey"]]


def load_noisy_labels(noisy_csv: str, source_name: str, canon_key_func):
    df = pd.read_csv(noisy_csv, low_memory=False)
    if SHA_COL not in df.columns or FAMILY_COL not in df.columns:
        raise RuntimeError(f"{noisy_csv} must contain '{SHA_COL}' and '{FAMILY_COL}'")

    df = df[[SHA_COL, FAMILY_COL]].copy()
    df[SHA_COL] = df[SHA_COL].astype(str).str.lower()
    df.rename(columns={FAMILY_COL: f"{source_name}_raw"}, inplace=True)

    norm_results = df[f"{source_name}_raw"].map(normalize_tool_family)
    df[f"{source_name}_family"] = norm_results.map(lambda x: normalize_family(x[0]) if x[0] is not None else None)
    df[f"{source_name}_status"] = norm_results.map(lambda x: x[1])
    df[f"{source_name}_is_labeled"] = df[f"{source_name}_status"] == "labeled"
    df[f"{source_name}_ckey"] = df[f"{source_name}_family"].map(canon_key_func)

    return df[[SHA_COL, f"{source_name}_family", f"{source_name}_ckey",
               f"{source_name}_status", f"{source_name}_is_labeled"]]


def build_joint_dataframe(features_csv: str, clean_csv: str, source_csvs: Dict[str, str], canon_key_func):
    feat_df, feature_cols = load_feature_matrix_csv(features_csv)
    clean_df = load_clean_labels(clean_csv, canon_key_func)
    df = feat_df.merge(clean_df, on=SHA_COL, how="inner")
    if df.empty:
        raise RuntimeError("No overlap between features and clean labels.")

    for src_name, src_csv in source_csvs.items():
        noisy_df = load_noisy_labels(src_csv, src_name, canon_key_func)
        df = df.merge(noisy_df, on=SHA_COL, how="left")

    print(f"[*] Joint dataframe before GT filtering: samples={len(df)}, features={len(feature_cols)}")
    return df, feature_cols


def filter_primary_eval_pool(df: pd.DataFrame, min_count: int = PRIMARY_MIN_GT_FAMILY_SIZE):
    counts = df["family_clean_ckey"].value_counts(dropna=False)
    keep_ckeys = set(counts[counts >= min_count].index)
    removed_ckeys = set(counts[counts < min_count].index)

    df_filtered = df[df["family_clean_ckey"].isin(keep_ckeys)].copy().reset_index(drop=True)

    print(f"[*] Primary eval filter: min GT family size = {min_count}")
    print(f"[*] Before filter: samples={len(df)}, GT families={df['family_clean_ckey'].nunique()}")
    print(f"[*] After  filter: samples={len(df_filtered)}, GT families={df_filtered['family_clean_ckey'].nunique()}")
    print(f"[*] Removed GT families: {len(removed_ckeys)}")
    return df_filtered, counts


def make_stratified_folds(y_clean_ckey, n_splits=5, seed=42):
    y_clean_ckey = np.asarray(y_clean_ckey)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    idx = np.arange(len(y_clean_ckey))
    folds = []
    for train_idx, test_idx in skf.split(idx, y_clean_ckey):
        folds.append((np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)))
    return folds


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


def avclass_cluster_metrics(y_true, y_pred):
    N = len(y_true)
    if N == 0:
        return 0.0, 0.0, 0.0

    pred_clusters = defaultdict(list)
    ref_clusters = defaultdict(list)

    for i in range(N):
        pred_clusters[y_pred[i]].append(i)
        ref_clusters[y_true[i]].append(i)

    pred_sets = {k: set(v) for k, v in pred_clusters.items()}
    ref_sets = {k: set(v) for k, v in ref_clusters.items()}

    prec_sum = 0
    for cj in pred_sets.values():
        prec_sum += max(len(cj & rk) for rk in ref_sets.values())

    rec_sum = 0
    for rk in ref_sets.values():
        rec_sum += max(len(cj & rk) for cj in pred_sets.values())

    prec = prec_sum / N
    rec = rec_sum / N
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    if not (0.0 <= prec <= 1.0 and 0.0 <= rec <= 1.0 and 0.0 <= f1 <= 1.0):
        raise RuntimeError(
            f"Cluster metrics out of range: prec={prec}, rec={rec}, f1={f1}. "
            f"Please check the normalization."
        )
    return prec, rec, f1


def build_model(input_dim, num_classes, device):
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    ).to(device)


def eval_fold(df, feature_cols, source_name, le, train_idx, test_idx, seed, device):
    if source_name == "gt":
        effective_train_idx = np.array(train_idx, dtype=int)
    else:
        labeled_mask = df.iloc[train_idx][f"{source_name}_is_labeled"].values
        effective_train_idx = np.array(train_idx[labeled_mask], dtype=int)

    if len(effective_train_idx) == 0:
        raise RuntimeError(f"[{source_name}] No labeled training samples in this fold.")

    effective_test_idx = np.array(test_idx, dtype=int)

    X_train = df.iloc[effective_train_idx][feature_cols].values.astype(np.float32)
    X_test = df.iloc[effective_test_idx][feature_cols].values.astype(np.float32)

    if source_name == "gt":
        y_train_str = df.iloc[effective_train_idx]["family_clean"].values
    else:
        y_train_str = df.iloc[effective_train_idx][f"{source_name}_family"].values
    y_test_str = df.iloc[effective_test_idx]["family_clean"].values

    y_train = le.transform(y_train_str)
    y_test = le.transform(y_test_str)

    train_class_to_idx = {c: np.flatnonzero(y_train == c) for c in np.unique(y_train)}

    set_seed(seed)
    input_dim = X_train.shape[1]
    num_classes = len(le.classes_)

    model = build_model(input_dim, num_classes, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=META_LR)

    prev_loss = float("inf")
    patience_cnt = 0

    for epoch in range(1, META_EPOCHS + 1):
        model.train()
        meta_loss = 0.0
        valid_tasks = 0

        denom = max(1, N_WAY * (K_SHOT + Q_QUERY))
        num_tasks = max(1, len(y_train) // denom)

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

        if abs(prev_loss - meta_loss.item()) < EPS_STOP:
            patience_cnt += 1
            if patience_cnt >= STOP_PATIENCE:
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

    for _ in range(FT_EPOCHS):
        for bx, by in ft_loader:
            bx, by = bx.to(device), by.to(device)
            pred = model(bx)
            loss = F.cross_entropy(pred, by)
            ft_opt.zero_grad()
            loss.backward()
            ft_opt.step()

    model.eval()
    with torch.no_grad():
        test_x = torch.from_numpy(X_test.astype(np.float32)).to(device)
        test_pred = model(test_x).argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_test, test_pred)
    mp = precision_score(y_test, test_pred, average="macro", zero_division=0)
    mr = recall_score(y_test, test_pred, average="macro", zero_division=0)
    mf1 = f1_score(y_test, test_pred, average="macro", zero_division=0)

    ami = adjusted_mutual_info_score(y_test, test_pred, average_method="arithmetic")
    c_prec, c_rec, c_f1 = avclass_cluster_metrics(y_test, test_pred)

    return {
        "n_train": len(effective_train_idx),
        "n_test": len(effective_test_idx),
        "acc": acc,
        "macro_precision": mp,
        "macro_recall": mr,
        "macro_f1": mf1,
        "ami": ami,
        "cluster_prec": c_prec,
        "cluster_rec": c_rec,
        "cluster_f1": c_f1,
    }


def run_source(df, feature_cols, source_name, folds, le, base_seed, out_dir, device):
    rows = []
    print(f"\n{'='*20} SOURCE: {source_name.upper()} {'='*20}")

    for fold_id, (train_idx, test_idx) in enumerate(folds, start=1):
        seed = base_seed + fold_id * 100
        metrics = eval_fold(df, feature_cols, source_name, le, train_idx, test_idx, seed, device)
        metrics["source"] = source_name
        metrics["fold"] = fold_id
        rows.append(metrics)

        print(
            f"Fold {fold_id}: "
            f"train={metrics['n_train']}, test={metrics['n_test']}, "
            f"ACC={metrics['acc']:.4f}, "
            f"Macro-Precision={metrics['macro_precision']:.4f}, "
            f"Macro-Recall={metrics['macro_recall']:.4f}, "
            f"Macro-F1={metrics['macro_f1']:.4f}, "
            f"AMI={metrics['ami']:.4f}, "
            f"C-Prec={metrics['cluster_prec']:.4f}, "
            f"C-Rec={metrics['cluster_rec']:.4f}, "
            f"C-F1={metrics['cluster_f1']:.4f}"
        )

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(os.path.join(out_dir, f"fold_metrics_{source_name}.csv"), index=False, encoding="utf-8-sig")

    summary = {
        "source": source_name,
        "n_folds": len(fold_df),
        "mean_train": fold_df["n_train"].mean(),
        "mean_test": fold_df["n_test"].mean(),
    }

    metric_cols = [
        "acc", "macro_precision", "macro_recall", "macro_f1",
        "ami", "cluster_prec", "cluster_rec", "cluster_f1"
    ]
    for m in metric_cols:
        summary[f"{m}_mean"] = fold_df[m].mean()
        summary[f"{m}_std"] = fold_df[m].std(ddof=0)
    return summary


def save_status_breakdown(df, source_name, out_dir):
    status_col = f"{source_name}_status"
    if status_col not in df.columns:
        return
    vc = df[status_col].value_counts(dropna=False).rename_axis("status").reset_index(name="count")
    vc["ratio"] = vc["count"] / len(df)
    vc.to_csv(os.path.join(out_dir, f"status_breakdown_{source_name}.csv"), index=False, encoding="utf-8-sig")


def save_primary_filter_info(df_before: pd.DataFrame, df_after: pd.DataFrame, out_dir: str):
    info = pd.DataFrame({
        "stat": ["samples_before", "samples_after", "gt_families_before", "gt_families_after"],
        "value": [
            len(df_before),
            len(df_after),
            df_before["family_clean_ckey"].nunique(),
            df_after["family_clean_ckey"].nunique(),
        ]
    })
    info.to_csv(os.path.join(out_dir, "primary_filter_info.csv"), index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--clean", required=True)
    parser.add_argument("--kaspersky", required=True)
    parser.add_argument("--avclass2", required=True)
    parser.add_argument("--claravy", required=True)
    parser.add_argument("--synonyms", default=None)
    parser.add_argument("--out", default="./out_metamamc_multi_primary")
    parser.add_argument("--device", default=DEVICE, choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    set_seed(SEED)
    device = pick_device(args.device)
    print(f"[*] Device: {device}")

    if args.synonyms:
        alias2canon_key, canon_key2display = load_synonyms(args.synonyms)
    else:
        alias2canon_key, canon_key2display = {}, {}
    _, canon_key_func = make_canonizer(alias2canon_key, canon_key2display)

    source_csvs = {
        "kaspersky": args.kaspersky,
        "avclass2": args.avclass2,
        "claravy": args.claravy,
    }

    df, feature_cols = build_joint_dataframe(args.features, args.clean, source_csvs, canon_key_func)
    df_before = df.copy()
    df, _ = filter_primary_eval_pool(df, min_count=PRIMARY_MIN_GT_FAMILY_SIZE)
    save_primary_filter_info(df_before, df, args.out)

    for src in source_csvs.keys():
        save_status_breakdown(df, src, args.out)

    all_labels = set(df["family_clean"].dropna().astype(str).tolist())
    for src in source_csvs.keys():
        all_labels.update(df[f"{src}_family"].dropna().astype(str).tolist())

    le = LabelEncoder()
    le.fit(sorted(all_labels))
    print(f"[*] Unified classes after primary filtering: {len(le.classes_)}")

    folds = make_stratified_folds(df["family_clean_ckey"].values, n_splits=N_FOLDS, seed=SEED)

    summaries = []
    for src in ["gt", "kaspersky", "avclass2", "claravy"]:
        summary = run_source(df, feature_cols, src, folds, le, SEED, args.out, device)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(os.path.join(args.out, "summary_all_sources.csv"), index=False, encoding="utf-8-sig")

    show_cols = [
        "source",
        "acc_mean", "acc_std",
        "macro_precision_mean", "macro_precision_std",
        "macro_recall_mean", "macro_recall_std",
        "macro_f1_mean", "macro_f1_std",
        "ami_mean", "ami_std",
        "cluster_prec_mean", "cluster_prec_std",
        "cluster_rec_mean", "cluster_rec_std",
        "cluster_f1_mean", "cluster_f1_std",
    ]
    print("\n===== FINAL SUMMARY =====")
    print(summary_df[show_cols].to_string(index=False))
    print(f"\n[OK] Results saved to: {args.out}")


if __name__ == "__main__":
    main()
