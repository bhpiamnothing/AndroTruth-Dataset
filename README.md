# AndroTruth: A Reliable Benchmark Android Malware Dataset Derived from Technical Expert Reports

## Overview

This repository accompanies the RAID 2026 paper *"AndroTruth: A Reliable Benchmark Android Malware Dataset Derived from Technical Expert Reports"*.

**Paper status:** accepted by **RAID 2026**.

**AndroTruth** is an Android malware family benchmark built from **traceable expert technical reports** rather than AV-consensus voting. The current release contains:

- **8,172 malware samples**
- **187 malware families**
- **276 security reports / blog posts**
- **42 vendors and individual analyst sources**
- a temporal span from **2016 to 2025**

The dataset is designed for research on **family-label reliability**, **temporal robustness**, and **multimodal Android malware analysis**.

The dataset is designed for research on **family-label reliability**, **temporal robustness**, and **multimodal Android malware analysis**.

## Why AndroTruth?

Most public Android malware datasets rely wholly or partly on VirusTotal engine aggregation for family labels. In contrast, AndroTruth prioritizes **label reliability over sheer scale**:

- family labels are derived from **expert technical reports**;
- each sample is linked to **auditable provenance metadata**;
- unresolved conflicts are **excluded rather than force-merged**;
- the release includes **APK access, static features, dynamic reports, and VirusTotal scan reports**.

During dataset construction, we extracted **8,421 unique hashes** after cross-report deduplication. Among them, **245 hashes could not be retrieved from Koodous** and were therefore excluded. An additional **4 samples** were removed because their family assignments remained unresolved after manual reconciliation. The final release contains **8,172 samples from 187 families**.

## Access to APK files



After the paper is accepted, we plan to make the APK files available to qualified researchers upon request. To ensure responsible and research-oriented use, access requests will be manually reviewed.

If you would like to request APK access after acceptance, please provide:

- your intended research purpose,
- your background in Android security / malware research,
- a link to your homepage, Google Scholar, or recent publications,
- if you are a student, a link to your advisor's academic profile,
- your Gmail address for permissioned APK access.

The official long-term contact email will be provided after peer review.

## Repository contents

### `AndroTruth.csv`

Core metadata file for the released samples. It provides per-sample provenance information, including:

- SHA-256 hash
- malware family
- source vendor / analyst
- first submission date (VirusTotal)
- report date
- report URL
- malware category

This file is the main entry point for auditing dataset provenance.

### `apks/`

Contains a subset of downloaded APK files. Full APK access is handled separately because of malware-distribution restrictions.

### `code_for_download_apks_from_koodous/`

Scripts used to:

- download APKs from Koodous,
- retrieve static and dynamic analysis reports from Koodous.

### `code_for_feature_extraction/`

Code for extracting Drebin-style static features from APKs.

We update the permission-to-API mapping so that feature extraction supports **Android API level 36**, improving coverage for modern Android malware samples.

### `dynamic_analysis_feature/`

Koodous dynamic analysis reports released with the dataset.

### `AndroTruth_Static_analysis/`

Released static features in Drebin-style format.

Static feature extraction was performed with **Androguard 4.1.3**. A small subset of samples could not be fully parsed during feature extraction.

### `malware_families_aliases.json`

Canonical malware-family alias file released with this version.

Format:

```json
[
  {
    "family": "adwind",
    "aliases": ["AlienSpy", "jRAT", "JBifrost"]
  },
  {
    "family": "anatsa",
    "aliases": ["TeaBot", "Toddler", "BankTeaBot"]
  }
]
```

This file is useful for:

- label harmonization,
- family-name normalization across reports,
- reproducible synonym handling in third-party studies.

### `conflict_table.csv`

Detailed information for the **4 excluded unresolved conflict samples**.

This file records, for each excluded sample:

- hash (SHA-256 or MD5 as available),
- conflicting label A / source A / URL A,
- conflicting label B / source B / URL B.

The current file includes the four excluded cases involving:

- **Anubis vs Hydra**
- **Youzicheng vs Cookiethief**
- **AmexTroll vs Ermac**
- **Wroba vs Xloader**

These samples are **not** part of the final 8,172-sample release.

### VirusTotal scan reports

This release also includes the full VirusTotal scan reports for all released samples.

These reports support:

- the detection-count analysis in the paper,
- the evaluation of AV-derived labels,
- future research on engine disagreement, confidence, and temporal drift.

### `Experiments/`

Reproducible experiment scripts for the paper.

We evaluate two representative probes:

- **AndMFC**
- **Meta-MAMC**

Because the original public implementations are incomplete or not directly runnable for our setting, the repository contains our **reimplemented and adapted versions** for reproducible comparison under multiple label sources.

---

## Experiment reproduction

## 1. Environment

Recommended environment:

- Python **3.9+**
- `numpy`
- `pandas`
- `scikit-learn`

Additionally required for **Meta-MAMC**:

- `torch`

Example installation:

```bash
pip install numpy pandas scikit-learn torch
```

## 2. Expected input files

Most experiment scripts expect CSV files with the following minimum columns.

### Feature matrix

A CSV with:

- `sha256`
- one or more numeric feature columns

### Clean ground-truth labels

A CSV with:

- `sha256`
- `family`

### Noisy label files

Separate CSV files for Kaspersky / AVClass2 / ClarAVy, each with:

- `sha256`
- `family`

### ClarAVy confidence file

A CSV with:

- `sha256` (or `a256` / `sha-256`)
- `family`
- `confidence`

### Metadata file (for temporal / zero-day appendix)

A CSV with:

- `sha256` (or another SHA-like column)
- `First Submission date`  
  (or a custom date column passed via `--date_col`)

---

## 3. Important note about the alias file

We release the family alias file as `malware_families_aliases.json` for transparency and for researchers who may want to reuse the alias mapping in their own data curation or label-harmonization pipeline.

**However, this file is not required to reproduce the experiments reported in our paper.** Before running the experiments, we have already applied synonym / alias normalization to the AV-derived label sources used in the paper, including **AVClass2**, **ClarAVy**, and **Kaspersky**. Therefore, the released experimental label files are already harmonized, and the paper's results can be reproduced **without loading any additional synonym / alias file**.

For reproducing the results in this paper, users can simply run the provided scripts with the released label files directly, **without providing `--synonyms`**.

The current experiment scripts still keep an optional `--synonyms` argument for users who want to apply alias normalization to newly collected labels or to custom external datasets. In those scripts, the optional synonym file is expected in a **plain-text line format** such as:

```text
canonical_family,alias1,alias2,alias3
```

If your own label files are already canonicalized, you may omit `--synonyms`.

---

## 4. Main-paper experiments

### 4.1 Primary downstream evaluation under real-world label noise

**AndMFC**

```bash
python AadMFC_Real_world_Label_Noise.py \
  --features path/to/features.csv \
  --clean path/to/gt_labels.csv \
  --kaspersky path/to/kaspersky_labels.csv \
  --avclass2 path/to/avclass2_labels.csv \
  --claravy path/to/claravy_labels.csv \
  --out out_andmfc_multi_primary
```

**Meta-MAMC**

```bash
python Meta_MAMC_Real_world_Label_Noise.py \
  --features path/to/features.csv \
  --clean path/to/gt_labels.csv \
  --kaspersky path/to/kaspersky_labels.csv \
  --avclass2 path/to/avclass2_labels.csv \
  --claravy path/to/claravy_labels.csv \
  --device auto \
  --out out_metamamc_multi_primary
```

Outputs include:

- `summary_all_sources.csv`
- `primary_filter_info.csv`
- `fold_metrics_<source>.csv`
- `status_breakdown_<source>.csv`

> Note: the AndMFC primary script filename is currently `AadMFC_Real_world_Label_Noise.py` in the repository and is kept here exactly as released.

---

### 4.2 Matched-pool decomposition

This experiment compares:

- `gt`
- `real_<source>`
- `synthetic_uniform`
- `synthetic_structured`

on the same source-specific matched pool.

**AndMFC**

```bash
python AndMFC_MatchedPool.py \
  --features path/to/features.csv \
  --clean path/to/gt_labels.csv \
  --kaspersky path/to/kaspersky_labels.csv \
  --avclass2 path/to/avclass2_labels.csv \
  --claravy path/to/claravy_labels.csv \
  --repeats 1 \
  --out out_andmfc_all_sources_matched_decomposition
```

**Meta-MAMC**

```bash
python Meta_MAMC_MatchedPool.py \
  --features path/to/features.csv \
  --clean path/to/gt_labels.csv \
  --kaspersky path/to/kaspersky_labels.csv \
  --avclass2 path/to/avclass2_labels.csv \
  --claravy path/to/claravy_labels.csv \
  --repeats 1 \
  --device auto \
  --out out_metamamc_all_sources_matched_decomposition
```

Outputs:

- `fold_metrics_all.csv`
- `summary_matched_pool_all_sources.csv`

---

### 4.3 ClarAVy confidence-threshold experiment

These scripts automatically evaluate the thresholds:

- `0.4`
- `0.5`
- `0.6`
- `0.7`

**AndMFC**

```bash
python AndMFC_GT_ClarAVy_confidence_batch.py \
  --features path/to/features.csv \
  --clean path/to/gt_labels.csv \
  --claravy_conf path/to/claravy_with_confidence.csv \
  --out out_andmfc_claravy_conf_batch
```

**Meta-MAMC**

```bash
python Meta_MAMC_GT_ClarAVy_confidence_batch.py \
  --features path/to/features.csv \
  --clean path/to/gt_labels.csv \
  --claravy_conf path/to/claravy_with_confidence.csv \
  --device auto \
  --out out_metamamc_claravy_conf_batch
```

Outputs:

- `summary_gt_claravy_confidence_all_thresholds.csv`
- per-threshold subdirectories containing:
  - `summary_gt_claravy_confidence.csv`
  - `claravy_confidence_pool_info.csv`
  - `fold_metrics_gt.csv`
  - `fold_metrics_claravy.csv`



## 5. Appendix experiments

### 5.1 Matched-sample singleton sensitivity

This appendix experiment keeps AVClass2 / ClarAVy singleton cases as a unified pseudo-label and forces all label sources to use the same train/test sample IDs.

**AndMFC**

```bash
python AndMFC_matched_singleton.py \
  --features path/to/features.csv \
  --clean path/to/gt_labels.csv \
  --kaspersky path/to/kaspersky_labels.csv \
  --avclass2 path/to/avclass2_labels.csv \
  --claravy path/to/claravy_labels.csv \
  --out out_andmfc_matched_singleton
```

**Meta-MAMC**

```bash
python Meta_MAMC_matched_singleton.py \
  --features path/to/features.csv \
  --clean path/to/gt_labels.csv \
  --kaspersky path/to/kaspersky_labels.csv \
  --avclass2 path/to/avclass2_labels.csv \
  --claravy path/to/claravy_labels.csv \
  --device auto \
  --out out_metamamc_matched_singleton
```

Outputs:

- `summary_all_sources.csv`
- `matched_pool_info.csv`
- `fold_metrics_<source>.csv`
- `status_breakdown_<source>.csv`

---

### 5.2 Temporal closed-set and zero-day family analysis

These appendix experiments evaluate **year-based temporal splits**.
In the commands below, the released scripts still use the argument name `--cutoffs`, but the values represent **split years**.

**AndMFC**

```bash
python AndMFC_temporal_zeroday.py \
  --features path/to/features.csv \
  --clean path/to/gt_labels.csv \
  --metadata path/to/AndroTruth.csv \
  --date_col "First Submission date" \
  --cutoffs 2020,2021,2022 \
  --out out_andmfc_temporal_zeroday
```

**Meta-MAMC**

```bash
python Meta_MAMC_temporal_zeroday.py \
  --features path/to/features.csv \
  --clean path/to/gt_labels.csv \
  --metadata path/to/AndroTruth.csv \
  --date_col "First Submission date" \
  --cutoffs 2020,2021,2022 \
  --device auto \
  --out out_metamamc_temporal_zeroday
```

Outputs include:

- `temporal_closedset_summary.csv`
- `zeroday_known_summary.csv`
- `zeroday_novel_summary.csv`

and per-year / per-setting CSV breakdowns where applicable.

---

## 6. Notes on reproducibility

- All evaluations use the released feature / label files aligned by `sha256`.
- The scripts globally remove GT families with very small sample counts before cross-validation when required by the protocol.
- The real-world label-noise experiments distinguish:
  - explicit family labels,
  - singleton / unresolved cases,
  - no-malicious-detection cases,
  - missing-scan cases.
- Meta-MAMC can run on CPU, but GPU is recommended for speed.

## Citation

If you use AndroTruth in your research, please cite our RAID 2026 paper:

> Hongpeng Bai, Yao Zhang, Minhong Dong, Shunzhe Zhao, Haobo Zhang, Lingyue Li, Yude Bai, Shuai Hu, and Guangquan Xu. *AndroTruth: A Reliable Benchmark Android Malware Dataset Derived from Technical Expert Reports*. RAID 2026, to appear.

```bibtex
@inproceedings{bai2026androtruth,
  title     = {AndroTruth: A Reliable Benchmark Android Malware Dataset Derived from Technical Expert Reports},
  author    = {Bai, Hongpeng and Zhang, Yao and Dong, Minhong and Zhao, Shunzhe and Zhang, Haobo and Li, Lingyue and Bai, Yude and Hu, Shuai and Xu, Guangquan},
  booktitle = {Proceedings of RAID 2026},
  year      = {2026},
  note      = {To appear}
}
```

## Contact

For dataset access, collaboration, or questions about the release, please contact **bai931214@tju.edu.cn**.