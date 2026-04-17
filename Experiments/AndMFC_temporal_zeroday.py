# -*- coding: utf-8 -*-
"""
Appendix experiments for AndMFC:
1) Temporal closed-set split
2) Zero-day family analysis

Metrics:
- ACC
- Macro-Precision
- Macro-Recall
- Macro-F1
- AMI
- Cluster-Precision / Recall / F1

Usage example:
python AndMFC_temporal_zeroday_appendix.py \
  --features AndroTruth_feature_3000_matrix_df.csv \
  --clean AndroTruth_labels.csv \
  --metadata AndroTruth.csv \
  --cutoffs 2020,2021,2022 \
  --out out_andmfc_temporal_zeroday
"""

import os
import re
import argparse
from collections import Counter, defaultdict
from typing import Dict, List

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
from sklearn.model_selection import StratifiedShuffleSplit


SHA_CANDIDATES = ["sha256", "a256", "sha-256", "hash", "sha_256"]
FAMILY_COL = "family"
TOPK = 1000
RF_N_ESTIMATORS = 200
BASE_SEED = 42
SVM_KW = dict(kernel="poly", gamma=0.1, C=1.0)
MIN_TRAIN_PER_FAMILY = 2


def normalize_family(x):
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
    # fallback: first column containing "hash"
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


def andmfc_pipeline(X_train, y_train, X_test, seed):
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=seed,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    k = min(TOPK, rf.feature_importances_.shape[0])
    top_idx = np.argsort(rf.feature_importances_)[-k:][::-1]

    svm = SVC(**SVM_KW)
    svm.fit(X_train[:, top_idx], y_train)
    return svm.predict(X_test[:, top_idx])


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


def run_temporal_closedset(df: pd.DataFrame, feature_cols: List[str], cutoff: int, out_dir: str, seed: int):
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

    le = LabelEncoder()
    le.fit(df.loc[final_train | final_test, "family_clean"])

    X_train = df.loc[final_train, feature_cols].values.astype(np.float32)
    X_test = df.loc[final_test, feature_cols].values.astype(np.float32)
    y_train = le.transform(df.loc[final_train, "family_clean"])
    y_test = le.transform(df.loc[final_test, "family_clean"])

    print(f"  Closed-set after filtering: train={final_train.sum()}, test={final_test.sum()}, families={len(le.classes_)}")

    y_pred = andmfc_pipeline(X_train, y_train, X_test, seed)
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


def run_zero_day(df: pd.DataFrame, feature_cols: List[str], cutoff: int, out_dir: str, seed: int):
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

    X_train = df.loc[final_train, feature_cols].values.astype(np.float32)
    y_train = le.transform(df.loc[final_train, "family_clean"])

    all_test_mask = known_test | novel_test
    if all_test_mask.sum() == 0:
        print("  [!] No future test samples.")
        return None, None

    X_all_test = df.loc[all_test_mask, feature_cols].values.astype(np.float32)
    y_pred_all = andmfc_pipeline(X_train, y_train, X_all_test, seed)

    all_test_indices = np.where(all_test_mask)[0]
    known_local = np.array([known_test[i] for i in all_test_indices])
    novel_local = np.array([novel_test[i] for i in all_test_indices])

    known_row = None
    if known_local.sum() > 0:
        y_known_true = le.transform(df.iloc[all_test_indices[known_local]]["family_clean"])
        y_known_pred = y_pred_all[known_local]
        known_metrics = compute_metrics(y_known_true, y_known_pred)
        print("[Known-Family Test Results]")
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
        print(f"[Novel-Family (Zero-Day) Analysis]")
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
    ap.add_argument("--out", default="./out_andmfc_temporal_zeroday")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df, feature_cols = load_data(args.features, args.clean, args.metadata, args.date_col)

    cutoffs = [int(x.strip()) for x in args.cutoffs.split(",") if x.strip()]

    temporal_rows = []
    known_rows = []
    novel_rows = []

    for cutoff in cutoffs:
        cutoff_dir = os.path.join(args.out, f"cutoff_{cutoff}")
        os.makedirs(cutoff_dir, exist_ok=True)

        temporal_row = run_temporal_closedset(df, feature_cols, cutoff, cutoff_dir, BASE_SEED + cutoff)
        if temporal_row is not None:
            temporal_rows.append(temporal_row)

        known_row, novel_row = run_zero_day(df, feature_cols, cutoff, cutoff_dir, BASE_SEED + 1000 + cutoff)
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

    print("\n[OK] AndMFC appendix experiments finished.")
    print(f"Results saved to: {args.out}")


if __name__ == "__main__":
    main()
