# -*- coding: utf-8 -*-
"""
Appendix experiments for Meta-MAMC:
1) Temporal closed-set split
2) Zero-day family analysis

Metrics:
- ACC
- Macro-Precision
- Macro-Recall
- Macro-F1
- AMI
- Cluster-Precision / Recall / F1
"""

import os
import re
import random
import argparse
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
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


SHA_CANDIDATES = ["sha256", "a256", "sha-256", "hash", "sha_256"]
FAMILY_COL = "family"

SEED = 42
DEVICE = "auto"

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

MIN_TRAIN_PER_FAMILY = 2
META_TRAIN_RATIO = 0.7


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


def find_sha_col(df: pd.DataFrame) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in SHA_CANDIDATES:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    for c in df.columns:
        if "hash" in c.lower():
            return c
    raise RuntimeError(f"No sha-like column found. Columns={list(df.columns)}")


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
    prec_sum = sum(max(len(cj & rk) for rk in ref_sets.values()) for cj in pred_sets.values())
    rec_sum = sum(max(len(cj & rk) for cj in pred_sets.values()) for rk in ref_sets.values())
    prec = prec_sum / N
    rec = rec_sum / N
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    mp = precision_score(y_true, y_pred, average="macro", zero_division=0)
    mr = recall_score(y_true, y_pred, average="macro", zero_division=0)
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    ami = adjusted_mutual_info_score(y_true, y_pred, average_method="arithmetic")
    c_p, c_r, c_f = avclass_cluster_metrics(y_true, y_pred)
    return {
        "acc": acc,
        "macro_precision": mp,
        "macro_recall": mr,
        "macro_f1": mf1,
        "ami": ami,
        "cluster_prec": c_p,
        "cluster_rec": c_r,
        "cluster_f1": c_f,
    }


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

    support_idx, support_y, query_idx, query_y = [], [], [], []
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

    support_idx, support_y, query_idx, query_y = [], [], [], []
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


def build_model(input_dim, num_classes, device):
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    ).to(device)


def meta_mamc_pipeline(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, seed: int, device):
    set_seed(seed)
    num_classes = len(np.unique(y_train))
    input_dim = X_train.shape[1]

    # internal split for meta-train and fine-tune
    try:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=META_TRAIN_RATIO, random_state=seed)
        meta_train_idx, ft_idx = next(sss.split(np.zeros(len(y_train)), y_train))
    except Exception:
        meta_train_idx = np.arange(len(y_train))
        ft_idx = np.arange(len(y_train))

    X_meta_train = X_train[meta_train_idx]
    y_meta_train = y_train[meta_train_idx]
    X_ft = X_train[ft_idx]
    y_ft = y_train[ft_idx]

    train_class_to_idx = {c: np.flatnonzero(y_meta_train == c) for c in np.unique(y_meta_train)}

    model = build_model(input_dim, num_classes, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=META_LR)

    prev_loss = float("inf")
    patience_cnt = 0

    for _epoch in range(1, META_EPOCHS + 1):
        model.train()
        meta_loss = 0.0
        valid_tasks = 0

        denom = max(1, N_WAY * (K_SHOT + Q_QUERY))
        num_tasks = max(1, len(y_meta_train) // denom)

        for _ in range(num_tasks):
            support_x, support_y, query_x, query_y = sample_task(
                train_class_to_idx, N_WAY, K_SHOT, Q_QUERY, P_MIX, X_meta_train, y_meta_train
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

        print(f"    [*] Meta Epoch {_epoch}/{META_EPOCHS}: Loss {meta_loss.item():.4f}")

        if abs(prev_loss - meta_loss.item()) < EPS_STOP:
            patience_cnt += 1
            if patience_cnt >= STOP_PATIENCE:
                print(f"    [*] Early stop at meta epoch {_epoch}")
                break
        else:
            patience_cnt = 0
        prev_loss = meta_loss.item()

    model.train()
    ft_opt = torch.optim.Adam(model.parameters(), lr=FT_LR)
    ft_dataset = TensorDataset(
        torch.from_numpy(X_ft.astype(np.float32)),
        torch.from_numpy(y_ft).long()
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
        print(f"    [*] FT Epoch {ft_epoch}/{FT_EPOCHS}: Loss {ft_loss / max(1, len(ft_loader)):.4f}")

    model.eval()
    with torch.no_grad():
        test_x = torch.from_numpy(X_test.astype(np.float32)).to(device)
        y_pred = model(test_x).argmax(dim=1).cpu().numpy()
    return y_pred


def load_data(features_csv: str, clean_csv: str, metadata_csv: str, date_col: str):
    feat_df = pd.read_csv(features_csv, low_memory=False)
    sha_col_feat = find_sha_col(feat_df)
    feat_df[sha_col_feat] = feat_df[sha_col_feat].astype(str).str.lower()
    feat_df = feat_df.rename(columns={sha_col_feat: "sha256"})

    feature_cols = [c for c in feat_df.columns if c != "sha256"]
    if not feature_cols:
        raise RuntimeError("No feature columns found.")
    feat_df[feature_cols] = feat_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    clean_df = pd.read_csv(clean_csv, low_memory=False)
    sha_col_clean = find_sha_col(clean_df)
    if FAMILY_COL not in clean_df.columns:
        raise RuntimeError(f"{clean_csv} must contain '{FAMILY_COL}'")
    clean_df = clean_df[[sha_col_clean, FAMILY_COL]].copy()
    clean_df = clean_df.rename(columns={sha_col_clean: "sha256", FAMILY_COL: "family_clean_raw"})
    clean_df["sha256"] = clean_df["sha256"].astype(str).str.lower()
    clean_df["family_clean"] = clean_df["family_clean_raw"].map(normalize_family)

    meta_df = pd.read_csv(metadata_csv, low_memory=False)
    sha_col_meta = find_sha_col(meta_df)
    if date_col not in meta_df.columns:
        raise RuntimeError(f"{metadata_csv} must contain date column '{date_col}'")
    meta_df = meta_df[[sha_col_meta, date_col]].copy()
    meta_df = meta_df.rename(columns={sha_col_meta: "sha256", date_col: "first_submission_date"})
    meta_df["sha256"] = meta_df["sha256"].astype(str).str.lower()
    meta_df["first_submission_date"] = pd.to_datetime(meta_df["first_submission_date"], errors="coerce")
    meta_df["year"] = meta_df["first_submission_date"].dt.year
    meta_df = meta_df.dropna(subset=["year"]).copy()
    meta_df["year"] = meta_df["year"].astype(int)

    df = feat_df.merge(clean_df[["sha256", "family_clean"]], on="sha256", how="inner") \
                .merge(meta_df[["sha256", "year"]], on="sha256", how="inner")

    print(f"[*] Loaded merged data: {len(df)} samples, {len(feature_cols)} features, "
          f"{df['family_clean'].nunique()} families, years {df['year'].min()}-{df['year'].max()}")
    return df, feature_cols




def print_metric_block(m, prefix=""):
    print(f"{prefix}ACC={m['acc']:.4f}  Macro-Precision={m['macro_precision']:.4f}  "
          f"Macro-Recall={m['macro_recall']:.4f}  Macro-F1={m['macro_f1']:.4f}")
    print(f"{prefix}AMI={m['ami']:.4f}  Cluster-Prec={m['cluster_prec']:.4f}  "
          f"Cluster-Rec={m['cluster_rec']:.4f}  Cluster-F1={m['cluster_f1']:.4f}")


def print_summary_table(title: str, rows, cols):
    if not rows:
        return
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    header = " ".join([f"{c:<18s}" for c in cols])
    print(header)
    print("-" * len(header))
    for row in rows:
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:<18.4f}")
            else:
                vals.append(f"{str(v):<18s}")
        print(" ".join(vals))


def save_dict_csv(path: str, records: List[Dict]):
    pd.DataFrame(records).to_csv(path, index=False, encoding="utf-8-sig")


def run_temporal_closedset(df: pd.DataFrame, feature_cols: List[str], cutoff: int, out_dir: str, seed: int, device):
    print("" + "=" * 70)
    print(f"TEMPORAL SPLIT: Train <= {cutoff}, Test > {cutoff}")
    print("=" * 70)

    train_mask = df["year"].values <= cutoff
    test_mask = df["year"].values > cutoff

    train_fams = set(df.loc[train_mask, "family_clean"])
    test_fams = set(df.loc[test_mask, "family_clean"])
    shared_fams = train_fams & test_fams

    print(f"  Train: {train_mask.sum()} samples, {len(train_fams)} families")
    print(f"  Test:  {test_mask.sum()} samples, {len(test_fams)} families")
    print(f"  Shared families: {len(shared_fams)}")

    shared_train = train_mask & df["family_clean"].isin(shared_fams).values
    shared_test = test_mask & df["family_clean"].isin(shared_fams).values

    train_counts = Counter(df.loc[shared_train, "family_clean"])
    ok_fams = {fam for fam, cnt in train_counts.items() if cnt >= MIN_TRAIN_PER_FAMILY}

    final_train = shared_train & df["family_clean"].isin(ok_fams).values
    final_test = shared_test & df["family_clean"].isin(ok_fams).values

    if final_train.sum() == 0 or final_test.sum() == 0:
        print(f"  [!] Not enough usable samples after filtering for cutoff {cutoff}.")
        return None

    train_labels = df.loc[final_train, "family_clean"].values
    test_labels = df.loc[final_test, "family_clean"].values

    le = LabelEncoder()
    le.fit(train_labels)
    y_train = le.transform(train_labels)
    y_test = le.transform(test_labels)

    X_train = df.loc[final_train, feature_cols].values.astype(np.float32)
    X_test = df.loc[final_test, feature_cols].values.astype(np.float32)

    print(f"  Closed-set after filtering: train={final_train.sum()}, test={final_test.sum()}, families={len(le.classes_)}")
    y_pred = meta_mamc_pipeline(X_train, y_train, X_test, seed, device)
    metrics = compute_metrics(y_test, y_pred)

    print(f"[Temporal Closed-set Results — cutoff {cutoff}]")
    print_metric_block(metrics, prefix="    ")

    per_year_rows = []
    test_years = df.loc[final_test, "year"].values
    print(f"[Per-Year Breakdown in Test Set]")
    has_year = False
    for yr in sorted(np.unique(test_years)):
        yr_mask = test_years == yr
        if yr_mask.sum() < 5:
            continue
        has_year = True
        yr_metrics = compute_metrics(y_test[yr_mask], y_pred[yr_mask])
        print(f"    {int(yr)}: n={yr_mask.sum():>5d}, ACC={yr_metrics['acc']:.4f}, Macro-F1={yr_metrics['macro_f1']:.4f}, AMI={yr_metrics['ami']:.4f}")
        per_year_rows.append({
            "cutoff": cutoff,
            "test_year": int(yr),
            "n": int(yr_mask.sum()),
            **yr_metrics
        })
    if not has_year:
        print("    [No test year has at least 5 samples after filtering]")
    if per_year_rows:
        save_dict_csv(os.path.join(out_dir, f"temporal_closedset_per_year_cutoff_{cutoff}.csv"), per_year_rows)

    row = {
        "cutoff": cutoff,
        "train_samples": int(final_train.sum()),
        "test_samples": int(final_test.sum()),
        "train_families": int(df.loc[final_train, "family_clean"].nunique()),
        "test_families": int(df.loc[final_test, "family_clean"].nunique()),
        **metrics,
    }
    return row


def run_zero_day(df: pd.DataFrame, feature_cols: List[str], cutoff: int, out_dir: str, seed: int, device):
    print("" + "=" * 70)
    print(f"ZERO-DAY EVALUATION: Cutoff = {cutoff}")
    print("=" * 70)

    train_mask = df["year"].values <= cutoff
    test_mask = df["year"].values > cutoff

    train_counts = Counter(df.loc[train_mask, "family_clean"])
    ok_train_fams = {fam for fam, cnt in train_counts.items() if cnt >= MIN_TRAIN_PER_FAMILY}
    final_train = train_mask & df["family_clean"].isin(ok_train_fams).values

    if final_train.sum() == 0:
        print("  [!] No usable training data.")
        return None, None

    known_fams = set(df.loc[final_train, "family_clean"])
    known_test = test_mask & df["family_clean"].isin(known_fams).values
    novel_test = test_mask & (~df["family_clean"].isin(known_fams)).values

    print(f"  Train: {final_train.sum()} samples, {len(known_fams)} families")
    print(f"  Test (known): {known_test.sum()} samples")
    print(f"  Test (novel): {novel_test.sum()} samples across {df.loc[novel_test, 'family_clean'].nunique()} families")

    le = LabelEncoder()
    le.fit(df.loc[final_train, "family_clean"])
    y_train = le.transform(df.loc[final_train, "family_clean"])
    X_train = df.loc[final_train, feature_cols].values.astype(np.float32)

    all_test_mask = known_test | novel_test
    if all_test_mask.sum() == 0:
        print("  [!] No future test samples.")
        return None, None

    X_all_test = df.loc[all_test_mask, feature_cols].values.astype(np.float32)
    y_pred_all = meta_mamc_pipeline(X_train, y_train, X_all_test, seed, device)

    all_test_indices = np.where(all_test_mask)[0]
    known_local = np.array([known_test[i] for i in all_test_indices])
    novel_local = np.array([novel_test[i] for i in all_test_indices])

    known_row = None
    if known_local.sum() > 0:
        y_known_true = le.transform(df.iloc[all_test_indices[known_local]]["family_clean"])
        y_known_pred = y_pred_all[known_local]
        known_metrics = compute_metrics(y_known_true, y_known_pred)
        print(f"[Known-Family Test Results]")
        print_metric_block(known_metrics, prefix="    ")
        known_row = {
            "cutoff": cutoff,
            "train_samples": int(final_train.sum()),
            "known_test_samples": int(known_local.sum()),
            "known_test_families": int(df.iloc[all_test_indices[known_local]]["family_clean"].nunique()),
            **known_metrics,
        }
    else:
        print("  [!] No known-family test samples.")

    novel_row = None
    if novel_local.sum() > 0:
        novel_true_str = df.iloc[all_test_indices[novel_local]]["family_clean"].values
        novel_pred_known = le.inverse_transform(y_pred_all[novel_local])

        novel_le = LabelEncoder()
        novel_true_enc = novel_le.fit_transform(novel_true_str)
        novel_pred_enc = y_pred_all[novel_local]

        c_p, c_r, c_f = avclass_cluster_metrics(novel_true_enc, novel_pred_enc)
        ami = adjusted_mutual_info_score(novel_true_enc, novel_pred_enc, average_method="arithmetic")
        dominant_family, dominant_count = Counter(novel_pred_known).most_common(1)[0]
        dominant_ratio = dominant_count / len(novel_pred_known)

        print(f"[Novel-Family (Zero-Day) Analysis]")
        print(f"    Novel samples: {len(novel_pred_known)}")
        print(f"    Distinct predicted known families: {len(set(novel_pred_known))}")
        print(f"    Dominant predicted known family: {dominant_family} ({dominant_ratio:.4f})")
        print(f"    AMI={ami:.4f}  Cluster-Prec={c_p:.4f}  Cluster-Rec={c_r:.4f}  Cluster-F1={c_f:.4f}")
        print(f"    Top confusion targets (novel -> predicted-as):")
        breakdown_rows = []
        for fam in sorted(set(novel_true_str)):
            fam_mask = novel_true_str == fam
            fam_preds = novel_pred_known[fam_mask]
            pred_counter = Counter(fam_preds)
            top_pred, top_cnt = pred_counter.most_common(1)[0]
            if fam_mask.sum() >= 3:
                print(f"      {fam:25s} ({fam_mask.sum():3d} samples) -> '{top_pred}' ({top_cnt}/{fam_mask.sum()})")
            breakdown_rows.append({
                "cutoff": cutoff,
                "novel_family": fam,
                "n_samples": int(fam_mask.sum()),
                "top_predicted_known_family": top_pred,
                "top_predicted_count": int(top_cnt),
                "top_predicted_ratio": float(top_cnt / fam_mask.sum()),
                "n_distinct_predicted_known_families": int(len(pred_counter)),
            })
        save_dict_csv(os.path.join(out_dir, f"zeroday_novel_breakdown_cutoff_{cutoff}.csv"), breakdown_rows)

        novel_row = {
            "cutoff": cutoff,
            "novel_test_samples": int(novel_local.sum()),
            "novel_test_families": int(len(set(novel_true_str))),
            "distinct_predicted_known_families": int(len(set(novel_pred_known))),
            "dominant_predicted_family_ratio": float(dominant_ratio),
            "ami": float(ami),
            "cluster_prec": float(c_p),
            "cluster_rec": float(c_r),
            "cluster_f1": float(c_f),
        }
    else:
        print("  [!] No novel-family zero-day test samples.")

    return known_row, novel_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--clean", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--date_col", default="First Submission date")
    ap.add_argument("--cutoffs", default="2020,2021,2022")
    ap.add_argument("--out", default="./out_metamamc_temporal_zeroday")
    ap.add_argument("--device", default=DEVICE, choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = pick_device(args.device)
    print(f"[*] Device: {device}")

    df, feature_cols = load_data(args.features, args.clean, args.metadata, args.date_col)
    cutoffs = [int(x.strip()) for x in args.cutoffs.split(",") if x.strip()]

    temporal_rows = []
    known_rows = []
    novel_rows = []

    for cutoff in cutoffs:
        cutoff_dir = os.path.join(args.out, f"cutoff_{cutoff}")
        os.makedirs(cutoff_dir, exist_ok=True)

        temporal_row = run_temporal_closedset(df, feature_cols, cutoff, cutoff_dir, SEED + cutoff, device)
        if temporal_row is not None:
            temporal_rows.append(temporal_row)

        known_row, novel_row = run_zero_day(df, feature_cols, cutoff, cutoff_dir, SEED + 1000 + cutoff, device)
        if known_row is not None:
            known_rows.append(known_row)
        if novel_row is not None:
            novel_rows.append(novel_row)

    if temporal_rows:
        save_dict_csv(os.path.join(args.out, "temporal_closedset_summary.csv"), temporal_rows)
        print_summary_table(
            "TEMPORAL CLOSED-SET SUMMARY",
            temporal_rows,
            ["cutoff", "train_samples", "test_samples", "acc", "macro_f1", "ami", "cluster_f1"]
        )
    if known_rows:
        save_dict_csv(os.path.join(args.out, "zeroday_known_summary.csv"), known_rows)
        print_summary_table(
            "ZERO-DAY KNOWN-FAMILY SUMMARY",
            known_rows,
            ["cutoff", "known_test_samples", "acc", "macro_f1", "ami", "cluster_f1"]
        )
    if novel_rows:
        save_dict_csv(os.path.join(args.out, "zeroday_novel_summary.csv"), novel_rows)
        print_summary_table(
            "ZERO-DAY NOVEL-FAMILY SUMMARY",
            novel_rows,
            ["cutoff", "novel_test_samples", "novel_test_families", "ami", "cluster_f1", "dominant_predicted_family_ratio"]
        )

    print("\n[OK] Meta-MAMC appendix experiments finished.")
    print(f"Results saved to: {args.out}")


if __name__ == "__main__":
    main()
