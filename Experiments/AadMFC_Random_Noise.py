
import numpy as np
import pandas as pd
import re
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold


# INPUT OUR ANDROTRUTH FEATURE CSV PATH HERE
FEATURES_CSV = r"Experiments\AndroTruth_feature_3000_matrix_df.csv"
# INPUT OUR ANDROTRUTH CLEAN LABEL CSV PATH HERE
CLEAN_LABELS_CSV = r"Experiments\AndroTruth_labels.csv"

SHA_COL = "sha256"
FAMILY_COL = "family"

N_FOLDS = 5
TOPK = 1000
RF_N_ESTIMATORS = 200
BASE_SEED = 42


SVM_KW = dict(kernel="poly", gamma=0.1, C=1.0)


MIN_TRAIN_PER_FAMILY = 1

# ---- Synthetic label noise ----
# Proportion of training labels to be randomly flipped
NOISE_RATIO = 0.05
NOISE_BY_CLASS = True




def normalize_family(x: str) -> str:
    """Normalize family name to avoid case/whitespace variants being treated as different classes."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.casefold()
    return s


def load_joint_pool(features_csv, clean_labels_csv):
    feat_df = pd.read_csv(features_csv)
    lab_df = pd.read_csv(clean_labels_csv)

    assert SHA_COL in feat_df.columns, f"{features_csv} must contain '{SHA_COL}'"
    assert SHA_COL in lab_df.columns and FAMILY_COL in lab_df.columns, \
        f"{clean_labels_csv} must contain '{SHA_COL}' and '{FAMILY_COL}'"

    lab_df = lab_df[[SHA_COL, FAMILY_COL]].copy()
    lab_df[FAMILY_COL] = lab_df[FAMILY_COL].map(normalize_family)

    df = feat_df.merge(lab_df, on=SHA_COL, how="inner")
    if df.empty:
        raise RuntimeError("No samples matched between features and labels by sha256.")

    feature_cols = [c for c in feat_df.columns if c != SHA_COL]
    if len(feature_cols) == 0:
        raise RuntimeError("No feature columns found (features file has only sha256?).")

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32).values
    y_str = df[FAMILY_COL].astype(str).values

    print(f"[*] Loaded: samples={len(df)}, features={len(feature_cols)}")
    print(f"[*] Unique families (normalized): {len(set(y_str))}")
    return X, y_str


def inject_random_label_noise(y, noise_ratio, n_classes, seed=42, by_class=True):
    """
    Randomly flip TRAIN labels only.
    - by_class=True: flip int(cnt * ratio) within each class (balanced)
    - by_class=False: flip int(N * ratio) globally
    Labels are flipped to a random different class id in [0, n_classes).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    y_noisy = y.copy()

    n = len(y_noisy)
    if n == 0 or noise_ratio <= 0:
        return y_noisy

    flip_indices = []

    if by_class:
        counts = Counter(y_noisy.tolist())
        for c, cnt in counts.items():
            n_flip = int(cnt * noise_ratio)
            if n_flip <= 0:
                continue
            idx_c = np.where(y_noisy == c)[0]
            n_flip = min(n_flip, len(idx_c))
            chosen = rng.choice(idx_c, size=n_flip, replace=False)
            flip_indices.extend(chosen.tolist())
    else:
        n_flip = int(n * noise_ratio)
        n_flip = max(0, min(n_flip, n))
        if n_flip > 0:
            flip_indices = rng.choice(np.arange(n), size=n_flip, replace=False).tolist()

    for idx in flip_indices:
        wrong = rng.integers(0, n_classes)
        while wrong == y_noisy[idx]:
            wrong = rng.integers(0, n_classes)
        y_noisy[idx] = wrong

    print(f"    [Synthetic Noise] flipped {len(flip_indices)}/{n} labels "
          f"({len(flip_indices)/max(n,1):.2%}), ratio={noise_ratio:.0%}, by_class={by_class}")
    return y_noisy


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


def eval_fold(X, y_clean_enc, le, train_idx, test_idx, seed):

    train_classes = set(y_clean_enc[train_idx])
    test_idx = np.array([i for i in test_idx if y_clean_enc[i] in train_classes], dtype=int)


    counts = Counter(y_clean_enc[train_idx])
    ok_classes = {c for c, cnt in counts.items() if cnt >= MIN_TRAIN_PER_FAMILY}

    train_idx = np.array([i for i in train_idx if y_clean_enc[i] in ok_classes], dtype=int)
    test_idx = np.array([i for i in test_idx if y_clean_enc[i] in ok_classes], dtype=int)

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError("After fold filtering, train/test became empty. "
                           "Try lowering MIN_TRAIN_PER_FAMILY.")

    X_tr, X_te = X[train_idx], X[test_idx]


    y_te = y_clean_enc[test_idx]


    y_tr_clean = y_clean_enc[train_idx]
    y_tr_noisy = inject_random_label_noise(
        y_tr_clean,
        noise_ratio=NOISE_RATIO,
        n_classes=len(le.classes_),
        seed=seed,
        by_class=NOISE_BY_CLASS
    )


    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=seed,
        n_jobs=-1
    )
    rf.fit(X_tr, y_tr_noisy)
    imp = rf.feature_importances_
    k = min(TOPK, imp.shape[0])
    top_idx = np.argsort(imp)[-k:][::-1]

    X_tr_k = X_tr[:, top_idx]
    X_te_k = X_te[:, top_idx]

    svm = SVC(**SVM_KW)
    svm.fit(X_tr_k, y_tr_noisy)
    y_pred = svm.predict(X_te_k)

    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_te, y_pred, average="macro", zero_division=0)

    top20_error_report(y_te, y_pred, le)
    return acc, f1, rec


def main():
    X, y_str = load_joint_pool(FEATURES_CSV, CLEAN_LABELS_CSV)

    le = LabelEncoder()
    le.fit(y_str)
    y_clean_enc = le.transform(y_str)

    print(f"[*] Encoded classes: {len(le.classes_)}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=BASE_SEED)

    all_acc, all_f1, all_rec = [], [], []
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(np.arange(len(y_clean_enc)), y_clean_enc), start=1):
        seed = BASE_SEED + fold_id * 100
        print(f"\n===== Fold {fold_id}/{N_FOLDS} (seed={seed}) =====")
        acc, f1, rec = eval_fold(X, y_clean_enc, le, train_idx, test_idx, seed)
        print(f"Fold {fold_id} -> Acc: {acc:.4f}, Macro-F1: {f1:.4f}, Macro-Recall: {rec:.4f}")
        all_acc.append(acc); all_f1.append(f1); all_rec.append(rec)

    print("\n===== 5-Fold CV Summary (train=synthetic-noisy, test=clean) =====")
    print(f"Acc mean/std: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"Macro-F1 mean/std: {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
    print(f"Macro-Recall mean/std: {np.mean(all_rec):.4f} ± {np.std(all_rec):.4f}")


if __name__ == "__main__":
    main()
