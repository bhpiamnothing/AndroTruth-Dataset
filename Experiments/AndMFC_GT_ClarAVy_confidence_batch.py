# -*- coding: utf-8 -*-
"""
Batch confidence-threshold evaluation for AndMFC on a GT + ClarAVy subset.

Fixed thresholds
----------------
This script automatically runs the same evaluation at:
    0.4, 0.5, 0.6, 0.7

No need to manually pass a specific threshold each time.

Pool construction for each threshold
------------------------------------
1. Align features, clean GT labels, and ClarAVy labels-with-confidence by sha256.
2. Keep only samples with:
   - explicit ClarAVy family label
   - confidence >= threshold
3. On the retained confidence-qualified pool, globally exclude GT families with fewer than 5 samples.
4. Run standard 5-fold stratified CV on the retained GT families.
5. Evaluate two training sources on the SAME filtered pool:
   - GT
   - ClarAVy

This ensures that every sample participating in training/testing satisfies the
ClarAVy confidence threshold, and that GT vs ClarAVy is compared fairly on the
same reduced subset.
"""

import os
import re
import argparse
from collections import defaultdict

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


SHA_CANDIDATES = ["sha256", "a256", "sha-256"]
FAMILY_COL = "family"
CONFIDENCE_COL = "confidence"

FIXED_THRESHOLDS = [0.4, 0.5, 0.6, 0.7]

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


def find_sha_col(df: pd.DataFrame) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in SHA_CANDIDATES:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    raise RuntimeError(f"No sha256-like column found. Expected one of {SHA_CANDIDATES}, got {list(df.columns)}")


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
    return prec, rec, f1


def load_features(features_csv: str):
    feat_df = pd.read_csv(features_csv, low_memory=False)
    sha_col = find_sha_col(feat_df)
    feat_df[sha_col] = feat_df[sha_col].astype(str).str.lower()
    feat_df.rename(columns={sha_col: "sha256"}, inplace=True)

    feature_cols = [c for c in feat_df.columns if c != "sha256"]
    if not feature_cols:
        raise RuntimeError("No feature columns found.")
    feat_df[feature_cols] = feat_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    return feat_df, feature_cols


def load_clean_labels(clean_csv: str):
    df = pd.read_csv(clean_csv, low_memory=False)
    sha_col = find_sha_col(df)
    if FAMILY_COL not in df.columns:
        raise RuntimeError(f"{clean_csv} must contain '{FAMILY_COL}'")
    df = df[[sha_col, FAMILY_COL]].copy()
    df.rename(columns={sha_col: "sha256", FAMILY_COL: "family_clean_raw"}, inplace=True)
    df["sha256"] = df["sha256"].astype(str).str.lower()
    df["family_clean"] = df["family_clean_raw"].map(normalize_family)
    return df[["sha256", "family_clean"]]


def load_claravy_with_confidence(claravy_csv: str):
    """
    Supports confidence stored either as:
      - ratio in [0, 1]
      - percentage in [0, 100]
    Internally we normalize to claravy_confidence_ratio in [0, 1].
    """
    df = pd.read_csv(claravy_csv, low_memory=False)
    sha_col = find_sha_col(df)
    required = {FAMILY_COL, CONFIDENCE_COL}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{claravy_csv} missing columns: {missing}")

    df = df[[sha_col, FAMILY_COL, CONFIDENCE_COL]].copy()
    df.rename(columns={sha_col: "sha256", FAMILY_COL: "claravy_family_raw", CONFIDENCE_COL: "claravy_confidence_raw"}, inplace=True)
    df["sha256"] = df["sha256"].astype(str).str.lower()
    df["claravy_confidence_raw"] = pd.to_numeric(df["claravy_confidence_raw"], errors="coerce")

    max_conf = df["claravy_confidence_raw"].dropna().max()
    if pd.notna(max_conf) and max_conf > 1.0:
        df["claravy_confidence_ratio"] = df["claravy_confidence_raw"] / 100.0
        conf_scale = "percent_to_ratio"
    else:
        df["claravy_confidence_ratio"] = df["claravy_confidence_raw"]
        conf_scale = "ratio"

    df["claravy_family"] = df["claravy_family_raw"].map(normalize_family)
    df["claravy_is_labeled"] = df["claravy_family"].notna() & (df["claravy_family"].astype(str).str.len() > 0)

    print(f"[*] ClarAVy confidence scale: {conf_scale}")
    return df


def build_confidence_pool(features_csv: str, clean_csv: str, claravy_csv: str, threshold: float):
    feat_df, feature_cols = load_features(features_csv)
    clean_df = load_clean_labels(clean_csv)
    claravy_df = load_claravy_with_confidence(claravy_csv)

    df = feat_df.merge(clean_df, on="sha256", how="inner").merge(claravy_df, on="sha256", how="inner")
    if df.empty:
        raise RuntimeError("No overlap between features, clean labels, and ClarAVy confidence file.")

    print(f"[*] Joint pool before confidence filter: samples={len(df)}, features={len(feature_cols)}")

    df["claravy_conf_ok"] = df["claravy_confidence_ratio"] >= threshold

    # Every participating sample must satisfy the confidence threshold and have an explicit ClarAVy family.
    df = df[df["claravy_is_labeled"] & df["claravy_conf_ok"]].copy().reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"No samples remain after applying the ClarAVy confidence threshold {threshold}.")

    print(f"[*] After confidence filter (confidence >= {threshold:.1f}): samples={len(df)}")
    return df, feature_cols


def filter_primary_eval_pool(df: pd.DataFrame, min_count: int = PRIMARY_MIN_GT_FAMILY_SIZE):
    counts = df["family_clean"].value_counts(dropna=False)
    keep = set(counts[counts >= min_count].index)
    removed = set(counts[counts < min_count].index)

    df_filtered = df[df["family_clean"].isin(keep)].copy().reset_index(drop=True)

    print(f"[*] Primary eval filter: min GT family size = {min_count}")
    print(f"[*] Before filter: samples={len(df)}, GT families={df['family_clean'].nunique()}")
    print(f"[*] After  filter: samples={len(df_filtered)}, GT families={df_filtered['family_clean'].nunique()}")
    print(f"[*] Removed GT families: {len(removed)}")
    return df_filtered


def make_stratified_folds(y_clean, n_splits=5, seed=42):
    y_clean = np.asarray(y_clean)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    idx = np.arange(len(y_clean))
    folds = []
    for train_idx, test_idx in skf.split(idx, y_clean):
        folds.append((np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)))
    return folds


def eval_fold(df, feature_cols, source_name, le, train_idx, test_idx, seed, topk=1000):
    effective_train_idx = np.array(train_idx, dtype=int)
    effective_test_idx = np.array(test_idx, dtype=int)

    X_tr = df.iloc[effective_train_idx][feature_cols].values.astype(np.float32)
    X_te = df.iloc[effective_test_idx][feature_cols].values.astype(np.float32)

    if source_name == "gt":
        y_tr_str = df.iloc[effective_train_idx]["family_clean"].values
    elif source_name == "claravy":
        y_tr_str = df.iloc[effective_train_idx]["claravy_family"].values
    else:
        raise RuntimeError(f"Unsupported source_name={source_name}")
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


def save_pool_info(df: pd.DataFrame, threshold: float, out_dir: str):
    info = pd.DataFrame({
        "stat": [
            "threshold",
            "samples_after_confidence_filter",
            "samples_after_gt_family_filter",
            "gt_families_after_gt_family_filter",
        ],
        "value": [
            threshold,
            len(df),
            len(df),
            df["family_clean"].nunique(),
        ],
    })
    info.to_csv(os.path.join(out_dir, "claravy_confidence_pool_info.csv"), index=False, encoding="utf-8-sig")


def run_one_threshold(features_csv: str, clean_csv: str, claravy_csv: str, threshold: float, out_root: str):
    thr_tag = f"thr_{int(threshold * 10):02d}"  # 0.4 -> thr_04
    out_dir = os.path.join(out_root, thr_tag)
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"[*] RUNNING THRESHOLD = {threshold:.1f}")
    print("=" * 80)

    df, feature_cols = build_confidence_pool(features_csv, clean_csv, claravy_csv, threshold)
    df = filter_primary_eval_pool(df, min_count=PRIMARY_MIN_GT_FAMILY_SIZE)
    save_pool_info(df, threshold, out_dir)

    all_labels = set(df["family_clean"].dropna().astype(str).tolist())
    all_labels.update(df["claravy_family"].dropna().astype(str).tolist())

    le = LabelEncoder()
    le.fit(sorted(all_labels))
    print(f"[*] Unified classes after confidence + GT filtering: {len(le.classes_)}")

    folds = make_stratified_folds(df["family_clean"].values, n_splits=N_FOLDS, seed=BASE_SEED)

    summaries = []
    for src in ["gt", "claravy"]:
        summary = run_source(df, feature_cols, src, folds, le, BASE_SEED, out_dir)
        summary["threshold"] = threshold
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(os.path.join(out_dir, "summary_gt_claravy_confidence.csv"), index=False, encoding="utf-8-sig")

    return summary_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--clean", required=True)
    parser.add_argument("--claravy_conf", required=True,
                        help="CSV with columns: sha256(or a256), family, confidence")
    parser.add_argument("--out", default="./out_andmfc_claravy_conf_batch")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    all_results = []
    for threshold in FIXED_THRESHOLDS:
        summary_df = run_one_threshold(
            args.features,
            args.clean,
            args.claravy_conf,
            threshold,
            args.out
        )
        all_results.append(summary_df)

    summary_all = pd.concat(all_results, axis=0, ignore_index=True)
    summary_all.to_csv(
        os.path.join(args.out, "summary_gt_claravy_confidence_all_thresholds.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 80)
    print("[OK] Batch run finished.")
    print(f"[*] Fixed thresholds: {FIXED_THRESHOLDS}")
    print(f"[*] Combined summary saved to: {os.path.join(args.out, 'summary_gt_claravy_confidence_all_thresholds.csv')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
