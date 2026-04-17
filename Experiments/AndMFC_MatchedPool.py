# -*- coding: utf-8 -*-
"""
Matched-pool decomposition experiment for AndMFC across multiple noisy sources:
  - Kaspersky
  - AVClass2
  - ClarAVy

For each source s, we define a source-specific shared pool D_s:
  GT label exists AND source s outputs an explicit family label.

Within each D_s:
1) Build a fixed 5-fold stratified split on GT labels.
2) On each fold, estimate source-vs-GT mismatch rate and confusion distribution
   using ONLY the training partition.
3) Compare four conditions on the exact same pool and split:
   - gt
   - real_<source>
   - synthetic_uniform
   - synthetic_structured

This yields an apples-to-apples decomposition of:
- shared-pool restriction
- corruption rate
- corruption structure

Outputs:
- fold_metrics_all.csv
- summary_matched_pool_all_sources.csv
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

SHA_COL = "sha256"
FAMILY_COL = "family"

N_FOLDS = 5
TOPK = 1000
RF_N_ESTIMATORS = 200
BASE_SEED = 42
PRIMARY_MIN_GT_FAMILY_SIZE = 5
DEFAULT_REPEATS = 1

SVM_KW = dict(kernel="poly", gamma=0.1, C=1.0)
SOURCE_ORDER = ["kaspersky", "avclass2", "claravy"]


def normalize_family(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s.casefold()


def keyize(s):
    s = str(s).lower().strip()
    return re.sub(r"[^a-z0-9]+", "", s)


def load_synonyms(path):
    alias2canon_key, canon_key2display = {}, {}
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
    return canon_key_func


def normalize_kaspersky_label(lbl):
    if lbl is None or (isinstance(lbl, float) and pd.isna(lbl)):
        return None, "missing"
    s = str(lbl).strip()
    s_lower = s.lower()
    if s == "" or s in {"-", "[]"}:
        return None, "missing"
    if s_lower == "n/a":
        return None, "missing_scan"
    if s_lower == "null":
        return None, "no_malicious_detection"
    return s, "labeled"


def normalize_avclass_claravy_label(lbl):
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
    prec_sum = sum(max(len(cj & rk) for rk in ref_sets.values()) for cj in pred_sets.values())
    rec_sum = sum(max(len(cj & rk) for cj in pred_sets.values()) for rk in ref_sets.values())
    prec = prec_sum / N
    rec = rec_sum / N
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def load_features(features_csv):
    feat_df = pd.read_csv(features_csv, low_memory=False)
    if SHA_COL not in feat_df.columns:
        raise RuntimeError(f"{features_csv} must contain '{SHA_COL}'")
    feature_cols = [c for c in feat_df.columns if c != SHA_COL]
    feat_df[SHA_COL] = feat_df[SHA_COL].astype(str).str.lower()
    feat_df[feature_cols] = feat_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    return feat_df, feature_cols


def load_clean_labels(clean_csv, canon_key_func):
    df = pd.read_csv(clean_csv, low_memory=False)
    df = df[[SHA_COL, FAMILY_COL]].copy()
    df[SHA_COL] = df[SHA_COL].astype(str).str.lower()
    df["family_clean"] = df[FAMILY_COL].astype(str).map(normalize_family)
    df["family_clean_ckey"] = df["family_clean"].map(canon_key_func)
    return df[[SHA_COL, "family_clean", "family_clean_ckey"]]


def load_tool_labels(tool_csv, source_name, canon_key_func):
    df = pd.read_csv(tool_csv, low_memory=False)
    df = df[[SHA_COL, FAMILY_COL]].copy()
    df[SHA_COL] = df[SHA_COL].astype(str).str.lower()
    df.rename(columns={FAMILY_COL: f"{source_name}_raw"}, inplace=True)

    if source_name == "kaspersky":
        norm_results = df[f"{source_name}_raw"].map(normalize_kaspersky_label)
    else:
        norm_results = df[f"{source_name}_raw"].map(normalize_avclass_claravy_label)

    df[f"{source_name}_family"] = norm_results.map(lambda x: normalize_family(x[0]) if x[0] is not None else None)
    df[f"{source_name}_status"] = norm_results.map(lambda x: x[1])
    df[f"{source_name}_ckey"] = df[f"{source_name}_family"].map(canon_key_func)

    return df[[SHA_COL, f"{source_name}_family", f"{source_name}_status", f"{source_name}_ckey"]]


def build_joint_dataframe(features_csv, clean_csv, tool_paths, canon_key_func):
    feat_df, feature_cols = load_features(features_csv)
    clean_df = load_clean_labels(clean_csv, canon_key_func)
    df = feat_df.merge(clean_df, on=SHA_COL, how="inner")
    if df.empty:
        raise RuntimeError("No overlap between features and clean labels.")

    for source_name, path in tool_paths.items():
        if path:
            tool_df = load_tool_labels(path, source_name, canon_key_func)
            df = df.merge(tool_df, on=SHA_COL, how="left")

    print(f"[*] Joint dataframe: samples={len(df)}, features={len(feature_cols)}")
    return df, feature_cols


def filter_to_source_matched_pool(df, source_name, min_count=PRIMARY_MIN_GT_FAMILY_SIZE):
    status_col = f"{source_name}_status"
    pool = df[df[status_col] == "labeled"].copy()
    counts = pool["family_clean_ckey"].value_counts(dropna=False)
    keep_ckeys = set(counts[counts >= min_count].index)
    pool = pool[pool["family_clean_ckey"].isin(keep_ckeys)].copy().reset_index(drop=True)
    print(f"[*] {source_name} shared pool: {len(pool)} samples, {pool['family_clean_ckey'].nunique()} GT families")
    return pool


def make_stratified_folds(y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(np.array(tr, dtype=int), np.array(te, dtype=int))
            for tr, te in skf.split(np.arange(len(y)), y)]


def estimate_training_confusion(train_df, source_name):
    family_col = f"{source_name}_family"
    mismatch_mask = train_df[family_col] != train_df["family_clean"]
    mismatch_rate = float(mismatch_mask.mean()) if len(train_df) else 0.0

    confused = train_df.loc[mismatch_mask, ["family_clean", family_col]]
    confusion_prob = {}
    for gt_fam, group in confused.groupby("family_clean"):
        vc = group[family_col].value_counts()
        total = vc.sum()
        confusion_prob[gt_fam] = {fam: cnt / total for fam, cnt in vc.items()}
    return mismatch_rate, confusion_prob


def inject_noise_matched(y_clean, mismatch_rate, mode, all_families, confusion_prob, rng):
    y_noisy = y_clean.copy()
    n = len(y_clean)
    n_flip = int(round(mismatch_rate * n))
    if n_flip <= 0:
        return y_noisy, 0

    flip_indices = rng.choice(n, size=n_flip, replace=False)
    n_actually_flipped = 0
    for idx in flip_indices:
        gt = y_clean[idx]
        if mode == "structured" and gt in confusion_prob and len(confusion_prob[gt]) > 0:
            targets = list(confusion_prob[gt].keys())
            probs = np.array([confusion_prob[gt][t] for t in targets], dtype=float)
            probs /= probs.sum()
            new_label = rng.choice(targets, p=probs)
        else:
            candidates = [f for f in all_families if f != gt]
            new_label = rng.choice(candidates)

        if new_label != gt:
            n_actually_flipped += 1
        y_noisy[idx] = new_label

    return y_noisy, n_actually_flipped


def eval_fold(X_tr, y_tr, X_te, y_te, le, seed):
    y_tr_enc = le.transform(y_tr)
    y_te_enc = le.transform(y_te)

    rf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=seed, n_jobs=-1)
    rf.fit(X_tr, y_tr_enc)
    imp = rf.feature_importances_
    k = min(TOPK, imp.shape[0])
    top_idx = np.argsort(imp)[-k:][::-1]

    svm = SVC(**SVM_KW)
    svm.fit(X_tr[:, top_idx], y_tr_enc)
    y_pred = svm.predict(X_te[:, top_idx])

    c_prec, c_rec, c_f1 = avclass_cluster_metrics(y_te_enc, y_pred)
    return {
        "acc":  accuracy_score(y_te_enc, y_pred),
        "macro_precision": precision_score(y_te_enc, y_pred, average="macro", zero_division=0),
        "macro_recall":    recall_score(y_te_enc, y_pred, average="macro", zero_division=0),
        "macro_f1":        f1_score(y_te_enc, y_pred, average="macro", zero_division=0),
        "ami":  adjusted_mutual_info_score(y_te_enc, y_pred, average_method="arithmetic"),
        "cluster_prec": c_prec,
        "cluster_rec":  c_rec,
        "cluster_f1":   c_f1,
    }


def run_one_source(pool, feature_cols, source_name, repeats):
    label_space = sorted(set(pool["family_clean"].dropna().unique().tolist()) |
                         set(pool[f"{source_name}_family"].dropna().unique().tolist()))
    le = LabelEncoder()
    le.fit(label_space)

    folds = make_stratified_folds(pool["family_clean_ckey"].values, n_splits=N_FOLDS, seed=BASE_SEED)
    results = []

    for fold_id, (train_idx, test_idx) in enumerate(folds, start=1):
        train_df = pool.iloc[train_idx].copy().reset_index(drop=True)
        test_df  = pool.iloc[test_idx].copy().reset_index(drop=True)

        X_tr = train_df[feature_cols].values.astype(np.float32)
        X_te = test_df[feature_cols].values.astype(np.float32)

        y_tr_gt = train_df["family_clean"].values
        y_tr_src = train_df[f"{source_name}_family"].values
        y_te_gt = test_df["family_clean"].values

        mismatch_rate, confusion_prob = estimate_training_confusion(train_df, source_name)
        all_families = sorted(train_df["family_clean"].dropna().unique().tolist())

        print(f"\n[{source_name}][Fold {fold_id}] train={len(train_df)} test={len(test_df)} "
              f"mismatch_rate={mismatch_rate:.4f} confusion_families={len(confusion_prob)}")

        for repeat in range(1, repeats + 1):
            seed = BASE_SEED + fold_id * 100 + repeat

            m_gt = eval_fold(X_tr, y_tr_gt, X_te, y_te_gt, le, seed)
            m_gt.update({
                "source": source_name,
                "condition": "gt",
                "fold": fold_id,
                "repeat": repeat,
                "train_samples": len(train_df),
                "test_samples": len(test_df),
                "mismatch_rate_train": mismatch_rate,
                "n_flipped": 0,
                "structured_confusion_families": len(confusion_prob),
            })
            results.append(m_gt)

            m_real = eval_fold(X_tr, y_tr_src, X_te, y_te_gt, le, seed)
            m_real.update({
                "source": source_name,
                "condition": f"real_{source_name}",
                "fold": fold_id,
                "repeat": repeat,
                "train_samples": len(train_df),
                "test_samples": len(test_df),
                "mismatch_rate_train": mismatch_rate,
                "n_flipped": int(round(mismatch_rate * len(train_df))),
                "structured_confusion_families": len(confusion_prob),
            })
            results.append(m_real)

            rng_u = np.random.RandomState(seed + 10000)
            y_u, n_flip_u = inject_noise_matched(
                y_tr_gt, mismatch_rate, "uniform", all_families, confusion_prob, rng_u
            )
            m_u = eval_fold(X_tr, y_u, X_te, y_te_gt, le, seed)
            m_u.update({
                "source": source_name,
                "condition": "synthetic_uniform",
                "fold": fold_id,
                "repeat": repeat,
                "train_samples": len(train_df),
                "test_samples": len(test_df),
                "mismatch_rate_train": mismatch_rate,
                "n_flipped": n_flip_u,
                "structured_confusion_families": len(confusion_prob),
            })
            results.append(m_u)

            rng_s = np.random.RandomState(seed + 20000)
            y_s, n_flip_s = inject_noise_matched(
                y_tr_gt, mismatch_rate, "structured", all_families, confusion_prob, rng_s
            )
            m_s = eval_fold(X_tr, y_s, X_te, y_te_gt, le, seed)
            m_s.update({
                "source": source_name,
                "condition": "synthetic_structured",
                "fold": fold_id,
                "repeat": repeat,
                "train_samples": len(train_df),
                "test_samples": len(test_df),
                "mismatch_rate_train": mismatch_rate,
                "n_flipped": n_flip_s,
                "structured_confusion_families": len(confusion_prob),
            })
            results.append(m_s)

            print(f"  repeat={repeat} | GT F1={m_gt['macro_f1']:.4f} | "
                  f"Real F1={m_real['macro_f1']:.4f} | "
                  f"U F1={m_u['macro_f1']:.4f} | "
                  f"S F1={m_s['macro_f1']:.4f}")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Drebin feature CSV")
    parser.add_argument("--clean", required=True, help="GT label CSV")
    parser.add_argument("--kaspersky", required=True, help="Kaspersky label CSV")
    parser.add_argument("--avclass2", required=True, help="AVClass2 label CSV")
    parser.add_argument("--claravy", required=True, help="ClarAVy label CSV")
    parser.add_argument("--synonyms", default=None, help="Family synonym file")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help="Independent repeats per fold (default: 1)")
    parser.add_argument("--out", default="./out_andmfc_all_sources_matched_decomposition")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.synonyms:
        alias2canon_key, canon_key2display = load_synonyms(args.synonyms)
    else:
        alias2canon_key, canon_key2display = {}, {}
    canon_key_func = make_canonizer(alias2canon_key, canon_key2display)

    tool_paths = {
        "kaspersky": args.kaspersky,
        "avclass2": args.avclass2,
        "claravy": args.claravy,
    }
    df, feature_cols = build_joint_dataframe(args.features, args.clean, tool_paths, canon_key_func)

    all_results = []
    for source_name in SOURCE_ORDER:
        pool = filter_to_source_matched_pool(df, source_name)
        src_results = run_one_source(pool, feature_cols, source_name, args.repeats)
        all_results.append(src_results)

    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(os.path.join(args.out, "fold_metrics_all.csv"), index=False, encoding="utf-8-sig")

    metric_cols = ["acc", "macro_precision", "macro_recall", "macro_f1",
                   "ami", "cluster_prec", "cluster_rec", "cluster_f1"]
    summary_rows = []
    for (source_name, cond), grp in results_df.groupby(["source", "condition"]):
        row = {
            "source": source_name,
            "condition": cond,
            "n_folds_x_repeats": len(grp),
            "mean_train_samples": grp["train_samples"].mean(),
            "mean_test_samples": grp["test_samples"].mean(),
            "matched_mismatch_rate_mean": grp["mismatch_rate_train"].mean(),
            "matched_n_flipped_mean": grp["n_flipped"].mean(),
            "structured_confusion_families_mean": grp["structured_confusion_families"].mean(),
        }
        for m in metric_cols:
            row[f"{m}_mean"] = grp[m].mean()
            row[f"{m}_std"] = grp[m].std(ddof=0)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    cond_order = {
        "gt": 0,
        "real_kaspersky": 1, "real_avclass2": 1, "real_claravy": 1,
        "synthetic_uniform": 2,
        "synthetic_structured": 3,
    }
    src_order = {k: i for i, k in enumerate(SOURCE_ORDER)}
    summary_df["__src"] = summary_df["source"].map(src_order)
    summary_df["__cond"] = summary_df["condition"].map(cond_order)
    summary_df = summary_df.sort_values(["__src", "__cond", "condition"]).drop(columns=["__src", "__cond"])
    summary_df.to_csv(os.path.join(args.out, "summary_matched_pool_all_sources.csv"), index=False, encoding="utf-8-sig")

    print("\n" + "=" * 88)
    print("SUMMARY: AndMFC matched-pool decomposition across Kaspersky / AVClass2 / ClarAVy")
    print("=" * 88)
    show_cols = ["source", "condition", "acc_mean", "macro_f1_mean", "ami_mean", "cluster_f1_mean",
                 "matched_mismatch_rate_mean", "matched_n_flipped_mean", "n_folds_x_repeats"]
    print(summary_df[show_cols].to_string(index=False))
    print(f"\n[OK] Results saved to: {args.out}")


if __name__ == "__main__":
    main()
