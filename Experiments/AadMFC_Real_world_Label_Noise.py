

import numpy as np
import pandas as pd
import re
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold

# ---------------- config ----------------
# INPUT OUR ANDROTRUTH FEATURE CSV PATH HERE
FEATURES_CSV = r"Experiments\AndroTruth_feature_3000_matrix_df.csv"
# INPUT KASPERSKY NOISY LABEL CSV PATH HERE, you can change to other noisy label sources if needed (e.g., avlcass2,claravy etc.) For GT labels experiment, please use AndroTruth_labels.csv
NOISY_LABELS_CSV = r"Experiments\AndroTruth_kaspersky_datasets.csv"
# INPUT OUR ANDROTRUTH CLEAN LABEL CSV PATH HERE
CLEAN_LABELS_CSV = r"Experiments\AndroTruth_labels.csv"

SHA_COL = "sha256"
FAMILY_COL = "family"

N_FOLDS = 5
TOPK = 1000
RF_N_ESTIMATORS = 200
BASE_SEED = 42

SVM_KW = dict(kernel="poly", gamma=0.1, C=1.0)


MIN_TRAIN_PER_FAMILY = 2
# ----------------------------------------


def normalize_family(x: str) -> str:

    if pd.isna(x):
        return ""
    s = str(x).strip()

    s = re.sub(r"\s+", " ", s)

    s = s.casefold()
    return s


def load_joint_pool(features_csv, noisy_labels_csv, clean_labels_csv):
    feat_df = pd.read_csv(features_csv)
    noisy_df = pd.read_csv(noisy_labels_csv)
    clean_df = pd.read_csv(clean_labels_csv)

    assert SHA_COL in feat_df.columns, f"{features_csv} must contain '{SHA_COL}'"
    assert SHA_COL in noisy_df.columns and FAMILY_COL in noisy_df.columns, \
        f"{noisy_labels_csv} must contain '{SHA_COL}' and '{FAMILY_COL}'"
    assert SHA_COL in clean_df.columns and FAMILY_COL in clean_df.columns, \
        f"{clean_labels_csv} must contain '{SHA_COL}' and '{FAMILY_COL}'"

    noisy_df = noisy_df[[SHA_COL, FAMILY_COL]].rename(columns={FAMILY_COL: "family_noisy"})
    clean_df = clean_df[[SHA_COL, FAMILY_COL]].rename(columns={FAMILY_COL: "family_clean"})

    # ✅ normalize family names on BOTH sides before merging/encoding
    noisy_df["family_noisy"] = noisy_df["family_noisy"].map(normalize_family)
    clean_df["family_clean"] = clean_df["family_clean"].map(normalize_family)

    # keep only samples that have BOTH labels
    df = feat_df.merge(noisy_df, on=SHA_COL, how="inner").merge(clean_df, on=SHA_COL, how="inner")
    if df.empty:
        raise RuntimeError("No samples have both noisy and clean labels after merging by sha256.")

    # feature columns
    feature_cols = [c for c in feat_df.columns if c != SHA_COL]
    if len(feature_cols) == 0:
        raise RuntimeError("No feature columns found (features file has only sha256?).")

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32).values
    y_noisy_str = df["family_noisy"].astype(str).values
    y_clean_str = df["family_clean"].astype(str).values

    print(f"[*] Joint pool: samples={len(df)}, features={len(feature_cols)}")
    return X, y_noisy_str, y_clean_str


def top20_error_report(y_te, y_pred, le):
    error_stats = []
    for c in np.unique(y_te):
        m = (y_te == c)
        n_total = int(m.sum())
        n_wrong = int((y_pred[m] != y_te[m]).sum())
        if n_total > 0:
            cname = le.inverse_transform([c])[0]
            error_stats.append((cname, n_total, n_wrong / n_total))

    top20 = sorted(error_stats, key=lambda x: x[2], reverse=True)[:20]
    print("\n  [Per-class error rate TOP-20] (test fold, clean labels)")
    print(f"  {'Class':25s} | {'#Samples':>8s} | {'Error rate':>10s}")
    print("  " + "-" * 50)
    for cname, n_total, err in top20:
        print(f"  {cname:25s} | {n_total:8d} | {err:10.2%}")


def eval_fold(X, y_noisy_enc, y_clean_enc, le, train_idx, test_idx, seed):

    train_clean_classes = set(y_clean_enc[train_idx])
    test_idx = np.array([i for i in test_idx if y_clean_enc[i] in train_clean_classes], dtype=int)

    counts = Counter(y_clean_enc[train_idx])
    ok_classes = {c for c, cnt in counts.items() if cnt >= MIN_TRAIN_PER_FAMILY}

    train_idx = np.array([i for i in train_idx if y_clean_enc[i] in ok_classes], dtype=int)
    test_idx = np.array([i for i in test_idx if y_clean_enc[i] in ok_classes], dtype=int)

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError("After fold filtering, train/test became empty. "
                           "Try lowering MIN_TRAIN_PER_FAMILY.")

    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr = y_noisy_enc[train_idx]
    y_te = y_clean_enc[test_idx]


    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=seed,
        n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    imp = rf.feature_importances_
    k = min(TOPK, imp.shape[0])
    top_idx = np.argsort(imp)[-k:][::-1]

    X_tr_k = X_tr[:, top_idx]
    X_te_k = X_te[:, top_idx]

    svm = SVC(**SVM_KW)
    svm.fit(X_tr_k, y_tr)
    y_pred = svm.predict(X_te_k)

    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_te, y_pred, average="macro", zero_division=0)

    top20_error_report(y_te, y_pred, le)
    return acc, f1, rec


def main():
    X, y_noisy_str, y_clean_str = load_joint_pool(FEATURES_CSV, NOISY_LABELS_CSV, CLEAN_LABELS_CSV)


    le = LabelEncoder()
    le.fit(np.concatenate([y_noisy_str, y_clean_str], axis=0))
    y_noisy_enc = le.transform(y_noisy_str)
    y_clean_enc = le.transform(y_clean_str)

    print(f"[*] Unified classes: {len(le.classes_)}")
    print(f"[*] Noisy==Clean agreement (raw string): {np.mean(y_noisy_str == y_clean_str):.2%}")


    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=BASE_SEED)

    all_acc, all_f1, all_rec = [], [], []
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(np.arange(len(y_clean_enc)), y_clean_enc), start=1):
        seed = BASE_SEED + fold_id * 100
        print(f"\n===== Fold {fold_id}/{N_FOLDS} (seed={seed}) =====")
        acc, f1, rec = eval_fold(X, y_noisy_enc, y_clean_enc, le, train_idx, test_idx, seed)
        print(f"Fold {fold_id} -> Acc: {acc:.4f}, Macro-F1: {f1:.4f}, Macro-Recall: {rec:.4f}")
        all_acc.append(acc); all_f1.append(f1); all_rec.append(rec)

    print("\n===== 5-Fold CV Summary (train=noisy, test=clean) =====")
    print(f"Acc mean/std: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"Macro-F1 mean/std: {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
    print(f"Macro-Recall mean/std: {np.mean(all_rec):.4f} ± {np.std(all_rec):.4f}")


if __name__ == "__main__":
    main()
