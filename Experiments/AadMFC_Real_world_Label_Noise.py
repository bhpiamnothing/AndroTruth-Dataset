# -*- coding: utf-8 -*-
"""
Primary downstream evaluation for AndMFC under multiple label sources:
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
import argparse
from collections import Counter, defaultdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    adjusted_mutual_info_score,
)
from sklearn.model_selection import StratifiedKFold


SHA_COL = "sha256"
FAMILY_COL = "family"

N_FOLDS = 5
TOPK = 1000
RF_N_ESTIMATORS = 200
BASE_SEED = 42
PRIMARY_MIN_GT_FAMILY_SIZE = 5

SVM_KW = dict(kernel="poly", gamma=0.1, C=1.0)


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


def load_synonyms(path: str):
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
    """
    Returns:
      (normalized_family_or_None, status)

    status:
      - labeled
      - missing_scan            : tool has no scan result (e.g., N/A)
      - no_malicious_detection  : tool scanned but did not flag as malware (e.g., null, benign)
      - singleton               : tool cannot assign a reliable family
      - missing                 : empty / malformed / unavailable value
    """
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


def load_features(features_csv: str):
    feat_df = pd.read_csv(features_csv, low_memory=False)
    if SHA_COL not in feat_df.columns:
        raise RuntimeError(f"{features_csv} must contain '{SHA_COL}'")
    feature_cols = [c for c in feat_df.columns if c != SHA_COL]
    if not feature_cols:
        raise RuntimeError("No feature columns found.")
    feat_df[SHA_COL] = feat_df[SHA_COL].astype(str).str.lower()
    feat_df[feature_cols] = feat_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    return feat_df, feature_cols


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
    feat_df, feature_cols = load_features(features_csv)
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


def eval_fold(df, feature_cols, source_name, le, train_idx, test_idx, seed, topk=1000):
    if source_name == "gt":
        effective_train_idx = np.array(train_idx, dtype=int)
    else:
        labeled_mask = df.iloc[train_idx][f"{source_name}_is_labeled"].values
        effective_train_idx = np.array(train_idx[labeled_mask], dtype=int)

    if len(effective_train_idx) == 0:
        raise RuntimeError(f"[{source_name}] No labeled training samples in this fold.")

    effective_test_idx = np.array(test_idx, dtype=int)

    X_tr = df.iloc[effective_train_idx][feature_cols].values.astype(np.float32)
    X_te = df.iloc[effective_test_idx][feature_cols].values.astype(np.float32)

    if source_name == "gt":
        y_tr_str = df.iloc[effective_train_idx]["family_clean"].values
    else:
        y_tr_str = df.iloc[effective_train_idx][f"{source_name}_family"].values
    y_te_str = df.iloc[effective_test_idx]["family_clean"].values

    y_tr = le.transform(y_tr_str)
    y_te = le.transform(y_te_str)

    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=seed,
        n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    imp = rf.feature_importances_
    k = min(topk, imp.shape[0])
    top_idx = np.argsort(imp)[-k:][::-1]

    X_tr_k = X_tr[:, top_idx]
    X_te_k = X_te[:, top_idx]

    svm = SVC(**SVM_KW)
    svm.fit(X_tr_k, y_tr)
    y_pred = svm.predict(X_te_k)

    acc = accuracy_score(y_te, y_pred)
    mp = precision_score(y_te, y_pred, average="macro", zero_division=0)
    mr = recall_score(y_te, y_pred, average="macro", zero_division=0)
    mf1 = f1_score(y_te, y_pred, average="macro", zero_division=0)

    ami = adjusted_mutual_info_score(y_te, y_pred, average_method="arithmetic")
    c_prec, c_rec, c_f1 = avclass_cluster_metrics(y_te, y_pred)

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


def run_source(df, feature_cols, source_name, folds, le, base_seed, out_dir):
    rows = []
    print(f"\n{'='*20} SOURCE: {source_name.upper()} {'='*20}")

    for fold_id, (train_idx, test_idx) in enumerate(folds, start=1):
        seed = base_seed + fold_id * 100
        metrics = eval_fold(df, feature_cols, source_name, le, train_idx, test_idx, seed)
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
    parser.add_argument("--out", default="./out_andmfc_multi_primary")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

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

    folds = make_stratified_folds(df["family_clean_ckey"].values, n_splits=N_FOLDS, seed=BASE_SEED)

    summaries = []
    for src in ["gt", "kaspersky", "avclass2", "claravy"]:
        summary = run_source(df, feature_cols, src, folds, le, BASE_SEED, args.out)
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
