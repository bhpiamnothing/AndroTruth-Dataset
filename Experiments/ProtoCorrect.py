#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProtoCorrect: Prototype-based Label Correction for Noisy Malware Family Classification

This script implements a prototype-based approach for training malware family classifiers
with noisy labels. It uses:
- Prototype learning with momentum updates
- Automatic label correction based on model confidence and prototype similarity
- Contrastive learning regularization
- FixMatch-style consistency regularization

Inputs:
  - Feature matrix CSV (drebin-style features)
  - Labels CSV with noisy family annotations

Output:
  - Corrected labels CSV with original and corrected family assignments
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter, defaultdict
from torch.optim.lr_scheduler import StepLR
import random
from sklearn.preprocessing import StandardScaler

# ==========================================
# 0. Reproducibility settings (must be at the top!)
# ==========================================
def set_seed(seed=42):
    """
    Set all random seeds to ensure fully reproducible results
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Key: Ensure deterministic computation
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✅ Random seed set to {seed} for reproducibility")

# Set random seed immediately
set_seed(42)

# ==========================================
# 1. Configuration and Hyperparameters
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURES_CSV_PATH = r"D:\Label_Correction\Datasets_Prepare\malradar_feature_3000_matrix_df.csv"
NOISY_LABELS_CSV_PATH = r"D:\Label_Correction\Datasets_Prepare\updated_malradar_avclass2_datasets_sha256.csv"

INPUT_DIM = 3000
HIDDEN_DIM = 512
BATCH_SIZE = 128
LR = 0.001
EPOCHS = 100
WARM_UP = 10
PROTOTYPE_MOMENTUM = 0.999
QUEUE_SIZE = 500
CORRECTION_START = 15
CONF_THRESHOLD = 0.95
WEAK_AUG_P = 0.9
STRONG_AUG_P = 0.8
BETA = 0.999
START_REWEIGHT_EPOCH = 20
TEMPERATURE = 0.07

# ==========================================
# 2. Basic Components
# ==========================================
def normalize_label_strict(name):
    if pd.isna(name):
        return "unknown"
    name = str(name).lower().strip()
    if len(name) == 0 or name == 'nan':
        return "unknown"
    return name

class ProtoMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, seed=42):
        super().__init__()
        # Fix weight initialization seed
        torch.manual_seed(seed)
        
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, hidden_dim)
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 128)
        )
        
        # Apply deterministic weight initialization
        self._init_weights()

    def _init_weights(self):
        """Apply fixed weight initialization strategy"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_embed=False):
        embed = self.backbone(x)
        logits = self.classifier(embed)
        if return_embed:
            proj = self.proj_head(embed)
            return logits, embed, proj
        return logits

class BaseDataset(Dataset):
    def __init__(self, X, y, indices):
        self.X = torch.FloatTensor(X[indices])
        self.y = torch.LongTensor(y[indices])
        self.indices = indices  # Global indices

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.indices[idx]

def augment_batch(x, p_a, seed_offset=0):
    """Deterministic data augmentation"""
    batch_size, feature_dim = x.shape
    device = x.device
    
    # Use hash-based seed for better determinism with reduced collision risk
    x_sum = torch.sum(x).item()
    # Combine sum with batch_size and feature_dim for more unique seeds
    seed_value = int(abs(hash((x_sum, batch_size, feature_dim, seed_offset))) % (2**31))
    torch.manual_seed(seed_value)
    mask = torch.bernoulli(torch.full((batch_size, feature_dim), p_a, device=device))
    
    # Deterministic permutation
    perm_indices = torch.randperm(batch_size, device=device)
    
    x_perm = x[perm_indices]
    return mask * x + (1 - mask) * x_perm

# ==========================================
# 3. Data Loading
# ==========================================
def load_data(features_path, labels_path):
    print(">>> Loading Data...")
    df_feat = pd.read_csv(features_path)
    df_noisy = pd.read_csv(labels_path)
    
    # Ensure consistent data loading order
    df_feat = df_feat.sort_values('sha256').reset_index(drop=True)
    df_noisy = df_noisy.sort_values('sha256').reset_index(drop=True)
    
    df = pd.merge(df_feat, df_noisy[['sha256', 'family']], on='sha256', how='left')
    
    df['is_singleton'] = df['family'].astype(str).str.strip().str.upper().str.startswith('SINGLETON:')
    singleton_count = df['is_singleton'].sum()
    print(f"  Identified {singleton_count} SINGLETON samples.")
    
    df['family_norm'] = df['family'].apply(normalize_label_strict)
    
    # Ensure consistent class mapping order
    fam_names = sorted(df['family_norm'].unique())
    fam2idx = {name: i for i, name in enumerate(fam_names)}
    idx2fam = {i: name for name, i in fam2idx.items()}
    df['label_code'] = df['family_norm'].map(fam2idx)
    
    ignore_cols = ['sha256', 'family', 'family_norm', 'label_code', 'is_singleton']
    feat_cols = [c for c in df.columns if c not in ignore_cols]
    
    # Ensure consistent feature column order
    feat_cols = sorted(feat_cols)
    
    X = df[feat_cols].values.astype(np.float32)
    X = StandardScaler().fit_transform(X)  # Standardize features
    y = df['label_code'].values
    singleton_mask_np = df['is_singleton'].values
    
    print(f"  Samples: {len(df)}, Classes: {len(fam_names)}")
    return X, y, singleton_mask_np, fam2idx, idx2fam, df

# ==========================================
# 4. Fixed ProtoCorrect Trainer
# ==========================================
class ProtoCorrectTrainer:
    def __init__(self, X, y, singleton_mask_np, num_classes, seed=42):
        self.X = X
        self.y = y
        self.num_classes = num_classes
        self.N = len(X)
        self.seed = seed
        
        # Set random seed for current training
        torch.manual_seed(seed)
        
        # Convert singleton_mask to torch Tensor
        self.singleton_mask = torch.tensor(singleton_mask_np, dtype=torch.bool, device=DEVICE)
        
        # Model initialization (deterministic init with seed in ProtoMLP)
        self.model = ProtoMLP(INPUT_DIM, HIDDEN_DIM, num_classes, seed=seed).to(DEVICE)
        
        # Optimizer (using fixed random state)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.5)
        
        self.class_weights = torch.ones(num_classes, device=DEVICE)
        
        # Prototypes and queues
        self.prototypes = torch.zeros(num_classes, HIDDEN_DIM, device=DEVICE)
        self.prototype_queues = defaultdict(list)
        
        # Record random state for each epoch
        self.epoch_seeds = {}

    def update_class_weights(self):
        # Simply based on original label distribution (can be changed to current clean samples)
        counts = Counter(self.y)
        weights = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            n = counts.get(c, 0)
            weights[c] = 0 if n == 0 else (1 - BETA ** n) / (1 - BETA)
        if weights.sum() > 0:
            weights = weights / weights.sum() * self.num_classes
        self.class_weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    def update_prototypes(self, embed, labels, batch_indices):
        with torch.no_grad():
            # Exclude singleton samples from current batch
            batch_singleton = self.singleton_mask[batch_indices]
            valid_mask = ~batch_singleton
            
            if valid_mask.sum() == 0:
                return
            
            embed_valid = embed[valid_mask]
            labels_valid = labels[valid_mask]
            
            for c in range(self.num_classes):
                mask_c = (labels_valid == c)
                if mask_c.sum() > 0:
                    class_embed = embed_valid[mask_c].mean(dim=0)
                    self.prototypes[c] = (
                        PROTOTYPE_MOMENTUM * self.prototypes[c] +
                        (1 - PROTOTYPE_MOMENTUM) * class_embed
                    )
                    
                    # Queue update (maintain deterministic order)
                    queue = self.prototype_queues[c]
                    queue.extend(embed_valid[mask_c].cpu().numpy())
                    if len(queue) > QUEUE_SIZE:
                        queue = queue[-QUEUE_SIZE:]
                    self.prototype_queues[c] = queue

    def prototype_similarity(self, embed):
        proto_norm = F.normalize(self.prototypes, dim=1)
        embed_norm = F.normalize(embed, dim=1)
        return torch.mm(embed_norm, proto_norm.t())

    def label_correction(self, logits, embed, orig_labels, batch_indices):
        probs = F.softmax(logits, dim=1)
        max_probs, pred_labels = torch.max(probs, dim=1)
        
        proto_sim = self.prototype_similarity(embed)
        proto_pred = torch.argmax(proto_sim, dim=1)
        
        # Correction condition: model prediction matches prototype but differs from original label
        # and confidence is high
        correction_mask = (
            (pred_labels == proto_pred) &
            (pred_labels != orig_labels) &
            (max_probs > CONF_THRESHOLD)
        )
        
        # Exclude singleton samples (don't correct them)
        singleton_batch = self.singleton_mask[batch_indices]
        correction_mask = correction_mask & ~singleton_batch
        
        corrected_labels = orig_labels.clone()
        corrected_labels[correction_mask] = pred_labels[correction_mask]
        return corrected_labels

    def contrastive_proto_loss(self, proj_embed, labels):
        proj_norm = F.normalize(proj_embed, dim=1)
        sim_matrix = torch.mm(proj_norm, proj_norm.t()) / TEMPERATURE
        mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        mask = mask - torch.eye(mask.shape[0], device=DEVICE)
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1e-8)
        loss = -(mask * F.log_softmax(sim_matrix, dim=1)).sum(dim=1) / mask_sum.squeeze()
        return loss.mean()

    def get_dataloader(self, epoch):
        """Create DataLoader with deterministic shuffle"""
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        
        # Set different but deterministic seed for each epoch
        epoch_seed = self.seed + epoch
        torch.manual_seed(epoch_seed)
        
        g = torch.Generator()
        g.manual_seed(epoch_seed)
        
        return DataLoader(
            BaseDataset(self.X, self.y, np.arange(self.N)), 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=g,
            num_workers=0  # Recommended to set to 0 for full determinism
        )

    def train_epoch(self, epoch):
        self.model.train()
        loader = self.get_dataloader(epoch)
        
        total_loss_s = total_loss_c = total_loss_u = 0.0
        steps = 0
        
        for x, y_orig, batch_indices in loader:
            # Set deterministic seed for each batch
            torch.manual_seed(self.seed + epoch * 1000 + steps)
            
            x, y_orig = x.to(DEVICE), y_orig.to(DEVICE)
            batch_indices = batch_indices.to(DEVICE)
            
            # Weak augmentation (using deterministic augmentation)
            x_weak = augment_batch(x, WEAK_AUG_P)
            logits, embed, proj = self.model(x_weak, return_embed=True)
            
            # Label correction
            if epoch >= CORRECTION_START:
                y = self.label_correction(logits, embed, y_orig, batch_indices)
            else:
                y = y_orig
            
            # Supervised loss
            loss_s = F.cross_entropy(
                logits, y,
                weight=self.class_weights if epoch >= START_REWEIGHT_EPOCH else None
            )
            
            # Contrastive regularization
            loss_c = self.contrastive_proto_loss(proj, y)
            
            # FixMatch consistency
            with torch.no_grad():
                x_weak_u = augment_batch(x, WEAK_AUG_P)
                logits_u_w, _, _ = self.model(x_weak_u, return_embed=True)
                probs_u = F.softmax(logits_u_w, dim=1)
                max_probs, pseudo = torch.max(probs_u, dim=1)
            
            x_strong = augment_batch(x, STRONG_AUG_P)
            logits_s, _, _ = self.model(x_strong, return_embed=True)
            mask = max_probs.ge(CONF_THRESHOLD).float()
            loss_u = (F.cross_entropy(logits_s, pseudo, reduction='none') * mask).mean()
            
            loss = loss_s + 0.5 * loss_c + loss_u
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (deterministic operation)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update prototypes (only with high confidence non-singleton samples)
            with torch.no_grad():
                high_conf_mask = (max_probs > CONF_THRESHOLD)
                clean_mask = high_conf_mask & ~self.singleton_mask[batch_indices]
                if clean_mask.sum() > 0:
                    self.update_prototypes(
                        embed[clean_mask], y[clean_mask], batch_indices[clean_mask]
                    )
            
            total_loss_s += loss_s.item()
            total_loss_c += loss_c.item()
            total_loss_u += loss_u.item()
            steps += 1
        
        return total_loss_s / steps, total_loss_c / steps, total_loss_u / steps

    def run(self):
        print("\n>>> Starting Fixed ProtoCorrect Training...")
        print(f"  Random Seed: {self.seed}")
        print(f"  Device: {DEVICE}")
        
        self.update_class_weights()  # Initial weighting
        
        # Warm-up
        for e in range(WARM_UP):
            ls, lc, lu = self.train_epoch(e)
            print(f"  Warm-up Epoch {e+1}/{WARM_UP} | Loss S:{ls:.4f} C:{lc:.4f} U:{lu:.4f}")
        
        # Main training
        for e in range(WARM_UP, EPOCHS):
            ls, lc, lu = self.train_epoch(e)
            self.scheduler.step()
            print(f"  Epoch {e+1}/{EPOCHS} | Loss S:{ls:.4f} C:{lc:.4f} U:{lu:.4f}")

    def predict_all(self):
        self.model.eval()
        # Use deterministic DataLoader for prediction
        loader = DataLoader(
            BaseDataset(self.X, self.y, np.arange(self.N)), 
            batch_size=BATCH_SIZE, 
            shuffle=False,
            num_workers=0  # Ensure prediction is also deterministic
        )
        preds = []
        with torch.no_grad():
            for x, _, _ in loader:
                logits = self.model(x.to(DEVICE))
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        return np.array(preds)

# ==========================================
# 5. Main Program
# ==========================================
def main():
    # Set global random seed
    SEED = 42
    set_seed(SEED)
    
    X, y, singleton_mask_np, fam2idx, idx2fam, df = load_data(
        FEATURES_CSV_PATH, NOISY_LABELS_CSV_PATH
    )
    
    # Use trainer with fixed seed
    trainer = ProtoCorrectTrainer(
        X, y, singleton_mask_np, num_classes=len(fam2idx), seed=SEED
    )
    trainer.run()
    
    final_preds = trainer.predict_all()
    
    df['corrected_label_code'] = final_preds
    df['corrected_family'] = df['corrected_label_code'].map(idx2fam)
    
    save_path = "MalProCor_results.csv"
    df[['sha256', 'family', 'corrected_family']].to_csv(save_path, index=False)
    
    # Save reproducibility information
    with open("reproducibility_info.txt", "w") as f:
        f.write(f"Model: ProtoCorrectTrainer\n")
        f.write(f"Random Seed: {SEED}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Parameters:\n")
        f.write(f"  EPOCHS: {EPOCHS}\n")
        f.write(f"  BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"  LR: {LR}\n")
        f.write(f"  CORRECTION_START: {CORRECTION_START}\n")
        f.write(f"  CONF_THRESHOLD: {CONF_THRESHOLD}\n")
        f.write(f"Results saved to: {save_path}\n")
    
    print(f"\n✅ Training completed! Results saved to: {save_path}")
    print(f"✅ Reproducibility info saved to: reproducibility_info.txt")
    
    # Verify reproducibility (optional)
    print(f"\n🔍 To verify reproducibility, run the script again with seed={SEED}")

if __name__ == "__main__":
    main()
