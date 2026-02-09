"""
DINOv2 SSL Fine-tuning Pipeline

This pipeline combines the best of both approaches:
1. Start with pre-trained DINOv2 (142M images of knowledge)
2. SSL fine-tune ONLY the last block(s) on CUB bird images
3. Supervised fine-tuning on the 20 competition classes

This gives domain adaptation to birds while keeping the powerful pre-trained features.
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Set seeds
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path("/gpfs/workdir/cardilloa/Marwa/Birds_Image_Classification")

# CUB YOLO-cropped images for SSL
CUB_CROPPED_DIR = PROJECT_ROOT / "DINO_Experiments/contrastive_results/cropped_images/cub"

# Competition data
COMP_BASE = PROJECT_ROOT / "bdma-07-competition-2026/BDMA7_project_files"
TRAIN_IMG_DIR = COMP_BASE / "train_images"
VAL_IMG_DIR = COMP_BASE / "val_images"
TEST_IMG_DIR = COMP_BASE / "test_images/mistery_cat"

# Cropped competition images
CROPPED_DIR = PROJECT_ROOT / "DINO_Experiments/contrastive_results/cropped_images"
TRAIN_CROPPED_DIR = CROPPED_DIR / "train"
VAL_CROPPED_DIR = CROPPED_DIR / "val"
TEST_CROPPED_DIR = CROPPED_DIR / "test"

OUTPUT_DIR = PROJECT_ROOT / "DINO_Experiments/dinov2_ssl_finetune_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Competition classes
COMPETITION_CLASSES = [
    "American_Crow", "Black_billed_Cuckoo", "Bobolink", "Brandt_Cormorant",
    "Brewer_Blackbird", "Bronzed_Cowbird", "Brown_Creeper", "Eastern_Towhee",
    "Fish_Crow", "Gray_Catbird", "Gray_crowned_Rosy_Finch", "Groove_billed_Ani",
    "Indigo_Bunting", "Lazuli_Bunting", "Painted_Bunting", "Red_winged_Blackbird",
    "Rusty_Blackbird", "Yellow_billed_Cuckoo", "Yellow_breasted_Chat",
    "Yellow_headed_Blackbird",
]

# CUB folders corresponding to competition classes (to exclude from SSL)
CUB_COMPETITION_FOLDERS = [
    "004.Groove_billed_Ani", "009.Brewer_Blackbird", "010.Red_winged_Blackbird",
    "011.Rusty_Blackbird", "012.Yellow_headed_Blackbird", "013.Bobolink",
    "014.Indigo_Bunting", "015.Lazuli_Bunting", "016.Painted_Bunting",
    "019.Gray_Catbird", "020.Yellow_breasted_Chat", "021.Eastern_Towhee",
    "023.Brandt_Cormorant", "026.Bronzed_Cowbird", "028.Brown_Creeper",
    "029.American_Crow", "030.Fish_Crow", "031.Black_billed_Cuckoo",
    "033.Yellow_billed_Cuckoo", "034.Gray_crowned_Rosy_Finch",
]

# Model config
DINOV2_MODEL = "dinov2_vits14"
IMG_SIZE = 518  # DINOv2 optimal
FEATURE_DIM = 384

# SSL config - ONLY fine-tune last block
SSL_EPOCHS = 30  # Much fewer epochs needed
SSL_BATCH_SIZE = 32
SSL_LR = 1e-5  # Very low LR for fine-tuning
SSL_UNFREEZE_BLOCKS = 1  # Only last block
SSL_TEMPERATURE = 0.1

# Supervised config
SUP_EPOCHS = 30
SUP_BATCH_SIZE = 32
SUP_LR = 1e-3
SUP_FINETUNE_LR = 1e-5
SUP_FOLDS = 5
SUP_UNFREEZE_EPOCH = 5
SUP_UNFREEZE_BLOCKS = 2
PATIENCE = 10


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


# ============================================================
# SSL DATASET (Contrastive pairs from CUB)
# ============================================================

def is_competition_species(folder_name: str) -> bool:
    return folder_name in CUB_COMPETITION_FOLDERS


class SSLContrastiveDataset(Dataset):
    """
    Dataset for contrastive SSL on CUB images.
    Excludes competition species to prevent data leakage.
    Returns two augmented views of each image.
    """
    
    def __init__(self, root_dir: Path):
        self.image_paths = []
        
        if not root_dir.exists():
            raise FileNotFoundError(f"CUB directory not found: {root_dir}")
        
        excluded = 0
        for class_dir in sorted(root_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            
            if is_competition_species(class_dir.name):
                excluded += 1
                continue
            
            for img_path in class_dir.glob("*.jpg"):
                self.image_paths.append(img_path)
        
        print(f"\n[SSL DATA]")
        print(f"  Total images: {len(self.image_paths)}")
        print(f"  Species excluded: {excluded}")
        
        # Strong augmentation for contrastive learning
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        # Return two different augmented views
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2


# ============================================================
# MODEL WITH PARTIAL FREEZING
# ============================================================

class DINOv2SSLModel(nn.Module):
    """
    DINOv2 model for SSL fine-tuning.
    Freezes everything except the last N blocks.
    """
    
    def __init__(self, unfreeze_blocks: int = 1):
        super().__init__()
        
        print(f"\n[MODEL] Loading pre-trained {DINOV2_MODEL}...")
        self.backbone = torch.hub.load('facebookresearch/dinov2', DINOV2_MODEL, pretrained=True)
        
        # Freeze all parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze last N blocks
        if hasattr(self.backbone, 'blocks'):
            total_blocks = len(self.backbone.blocks)
            print(f"  Total transformer blocks: {total_blocks}")
            print(f"  Unfreezing last {unfreeze_blocks} blocks for SSL")
            
            for block in self.backbone.blocks[-unfreeze_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
        
        # Projection head for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(FEATURE_DIM, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        )
        
        self._log_params()
    
    def _log_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total params: {total/1e6:.2f}M")
        print(f"  Trainable params: {trainable/1e6:.2f}M ({100*trainable/total:.1f}%)")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (features, projections)"""
        features = self.backbone(x)  # [B, 384]
        projections = self.projector(features)  # [B, 256]
        projections = F.normalize(projections, dim=-1)  # L2 normalize
        return features, projections
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get features without projection (for downstream tasks)"""
        return self.backbone(x)


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    NT-Xent (InfoNCE) contrastive loss.
    z1, z2: [B, D] normalized embeddings of two views
    """
    batch_size = z1.size(0)
    
    # Concatenate views
    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    
    # Compute similarity matrix
    sim = torch.mm(z, z.t()) / temperature  # [2B, 2B]
    
    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, float('-inf'))
    
    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z.device),
        torch.arange(0, batch_size, device=z.device)
    ])
    
    loss = F.cross_entropy(sim, labels)
    return loss


# ============================================================
# PHASE 1: SSL FINE-TUNING
# ============================================================

def train_ssl():
    """SSL fine-tune the last block of DINOv2 on bird images."""
    
    print("\n" + "="*70)
    print("PHASE 1: SSL FINE-TUNING ON CUB BIRDS")
    print("="*70)
    
    # Dataset
    dataset = SSLContrastiveDataset(CUB_CROPPED_DIR)
    loader = DataLoader(
        dataset, batch_size=SSL_BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=(DEVICE.type == "cuda"), drop_last=True
    )
    
    # Model
    model = DINOv2SSLModel(unfreeze_blocks=SSL_UNFREEZE_BLOCKS).to(DEVICE)
    
    # Optimizer - only trainable params
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=SSL_LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SSL_EPOCHS)
    
    history = []
    best_loss = float('inf')
    
    for epoch in range(SSL_EPOCHS):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(loader, desc=f"[SSL] Epoch {epoch+1}/{SSL_EPOCHS}")
        for view1, view2 in pbar:
            view1 = view1.to(DEVICE, non_blocking=True)
            view2 = view2.to(DEVICE, non_blocking=True)
            
            # Forward
            _, z1 = model(view1)
            _, z2 = model(view2)
            
            # Contrastive loss
            loss = contrastive_loss(z1, z2, temperature=SSL_TEMPERATURE)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        avg_loss = total_loss / len(loader)
        lr = optimizer.param_groups[0]['lr']
        
        print(f"[SSL] Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
        
        history.append({'epoch': epoch+1, 'loss': avg_loss, 'lr': lr})
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'backbone_state': model.backbone.state_dict(),
                'projector_state': model.projector.state_dict(),
                'loss': best_loss,
            }, OUTPUT_DIR / "ssl_best.pth")
            print(f"  ✨ New best loss: {best_loss:.4f}")
        
        # Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'backbone_state': model.backbone.state_dict(),
                'projector_state': model.projector.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'history': history,
            }, OUTPUT_DIR / f"ssl_epoch{epoch+1:03d}.pth")
    
    # Save history
    with open(OUTPUT_DIR / "ssl_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n[SSL] Training complete! Best loss: {best_loss:.4f}")
    
    # Quick evaluation with linear probe
    print("\n[SSL] Running linear probe evaluation...")
    evaluate_ssl_features(model)
    
    return model


def evaluate_ssl_features(model: DINOv2SSLModel):
    """Quick linear probe to evaluate SSL feature quality."""
    
    # Load competition data
    df = load_competition_data()
    
    # Simple transform
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Extract features
    model.eval()
    features, labels = [], []
    
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
            img = Image.open(row['path']).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            feat = model.get_features(img_tensor).cpu().numpy()[0]
            features.append(feat)
            labels.append(row['label'])
    
    features = np.array(features)
    labels = np.array(labels)
    
    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Standardize and train logistic regression
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    
    train_acc = clf.score(X_train, y_train) * 100
    test_acc = clf.score(X_test, y_test) * 100
    
    print(f"\n[LINEAR PROBE] Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")
    
    return test_acc


# ============================================================
# PHASE 2: SUPERVISED FINE-TUNING
# ============================================================

class BirdDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        label = int(row["label"])
        if self.transform:
            img = self.transform(img)
        return img, label


def load_competition_data() -> pd.DataFrame:
    """Load competition train + val data."""
    
    def gather(base_dir: Path) -> List[dict]:
        rows = []
        if not base_dir.exists():
            return rows
        for class_dir in sorted(base_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            if class_dir.name not in COMPETITION_CLASSES:
                continue
            for img in class_dir.glob("*.jpg"):
                rows.append({"path": str(img), "class_name": class_dir.name})
        return rows
    
    # Use cropped if available
    if TRAIN_CROPPED_DIR.exists():
        rows = gather(TRAIN_CROPPED_DIR) + gather(VAL_CROPPED_DIR)
        print(f"  Using YOLO-cropped images")
    else:
        rows = gather(TRAIN_IMG_DIR) + gather(VAL_IMG_DIR)
        print(f"  Using original images")
    
    df = pd.DataFrame(rows)
    cls_to_idx = {name: i for i, name in enumerate(sorted(COMPETITION_CLASSES))}
    df["label"] = df["class_name"].map(cls_to_idx).astype(int)
    
    print(f"\n[DATA] {len(df)} images, {len(cls_to_idx)} classes")
    return df


def get_transforms(is_train: bool):
    if is_train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 2.0))], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class DINOv2Classifier(nn.Module):
    """Classifier using SSL fine-tuned DINOv2 backbone."""
    
    def __init__(self, num_classes: int = 20, dropout: float = 0.3):
        super().__init__()
        
        # Load pre-trained DINOv2
        print(f"\n[MODEL] Loading {DINOV2_MODEL}...")
        self.backbone = torch.hub.load('facebookresearch/dinov2', DINOV2_MODEL, pretrained=True)
        
        # Try to load SSL fine-tuned weights
        ssl_ckpt = OUTPUT_DIR / "ssl_best.pth"
        if ssl_ckpt.exists():
            print(f"  Loading SSL fine-tuned weights from {ssl_ckpt}")
            ckpt = torch.load(ssl_ckpt, map_location='cpu')
            self.backbone.load_state_dict(ckpt['backbone_state'])
            print(f"  ✅ Loaded SSL weights (loss: {ckpt.get('loss', 'N/A')})")
        else:
            print(f"  ⚠️ No SSL checkpoint found, using vanilla pre-trained")
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(FEATURE_DIM),
            nn.Dropout(dropout),
            nn.Linear(FEATURE_DIM, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )
        
        self._log_params()
    
    def _log_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total: {total/1e6:.2f}M | Trainable: {trainable/1e6:.2f}M ({100*trainable/total:.1f}%)")
    
    def unfreeze_last_blocks(self, n: int):
        if hasattr(self.backbone, 'blocks'):
            print(f"\n[UNFREEZE] Last {n} blocks")
            for block in self.backbone.blocks[-n:]:
                for param in block.parameters():
                    param.requires_grad = True
            self._log_params()
    
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


def train_supervised():
    """Supervised fine-tuning with 5-fold CV."""
    
    print("\n" + "="*70)
    print("PHASE 2: SUPERVISED FINE-TUNING")
    print("="*70)
    
    df = load_competition_data()
    X, y = df["path"].values, df["label"].values
    
    skf = StratifiedKFold(n_splits=SUP_FOLDS, shuffle=True, random_state=42)
    results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{SUP_FOLDS}")
        print(f"{'='*60}")
        
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        print(f"  Train: {len(train_df)} | Val: {len(val_df)}")
        
        train_ds = BirdDataset(train_df, get_transforms(True))
        val_ds = BirdDataset(val_df, get_transforms(False))
        
        train_loader = DataLoader(train_ds, SUP_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, SUP_BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        
        # Model
        model = DINOv2Classifier(num_classes=len(COMPETITION_CLASSES)).to(DEVICE)
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=SUP_LR, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, SUP_EPOCHS)
        
        best_acc = 0.0
        best_state = None
        patience_counter = 0
        
        for epoch in range(SUP_EPOCHS):
            # Gradual unfreezing
            if epoch == SUP_UNFREEZE_EPOCH:
                model.unfreeze_last_blocks(SUP_UNFREEZE_BLOCKS)
                optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=SUP_FINETUNE_LR, weight_decay=1e-4
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, SUP_EPOCHS - epoch)
            
            # Train
            model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
                train_correct += (logits.argmax(1) == labels).sum().item()
                train_total += labels.size(0)
            
            # Validate
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Val", leave=False):
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    logits = model(imgs)
                    val_correct += (logits.argmax(1) == labels).sum().item()
                    val_total += labels.size(0)
            
            scheduler.step()
            
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            print(f"[Fold {fold+1}] Epoch {epoch+1:02d} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                print(f"  ✨ New best: {best_acc:.2f}%")
            else:
                patience_counter += 1
            
            if patience_counter >= PATIENCE:
                print(f"  ⏹️ Early stopping")
                break
        
        # Save
        torch.save(best_state, OUTPUT_DIR / f"fold{fold+1}_best.pth")
        print(f"\n💾 Fold {fold+1} best: {best_acc:.2f}%")
        results.append(best_acc)
    
    # Summary
    print(f"\n{'='*60}")
    print("CV SUMMARY")
    print(f"{'='*60}")
    for i, acc in enumerate(results):
        print(f"  Fold {i+1}: {acc:.2f}%")
    print(f"  Mean: {np.mean(results):.2f}% ± {np.std(results):.2f}%")
    
    with open(OUTPUT_DIR / "cv_summary.json", 'w') as f:
        json.dump({'folds': results, 'mean': np.mean(results), 'std': np.std(results)}, f)


# ============================================================
# PHASE 3: SUBMISSION
# ============================================================

def tta_predict(model, img: Image.Image, device) -> torch.Tensor:
    """TTA prediction for a single image."""
    
    base_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    variants = [
        base_tf(img),
        base_tf(img.transpose(Image.FLIP_LEFT_RIGHT)),
    ]
    for angle in [-10, -5, 5, 10]:
        variants.append(base_tf(img.rotate(angle)))
    
    batch = torch.stack(variants).to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = F.softmax(logits, dim=1).mean(dim=0)
    return probs


def generate_submission():
    """Generate submission with ensemble + TTA."""
    
    print("\n" + "="*70)
    print("GENERATING SUBMISSION")
    print("="*70)
    
    cls_to_idx = {n: i for i, n in enumerate(sorted(COMPETITION_CLASSES))}
    idx_to_cls = {v: k for k, v in cls_to_idx.items()}
    
    test_dir = TEST_CROPPED_DIR if TEST_CROPPED_DIR.exists() else TEST_IMG_DIR
    test_paths = sorted(test_dir.glob("*.jpg"))
    print(f"[SUBMIT] {len(test_paths)} test images from {test_dir}")
    
    # Load fold models
    models = []
    for fold in range(SUP_FOLDS):
        ckpt = OUTPUT_DIR / f"fold{fold+1}_best.pth"
        if not ckpt.exists():
            continue
        model = DINOv2Classifier(len(COMPETITION_CLASSES)).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()
        models.append(model)
        print(f"  ✅ Loaded fold {fold+1}")
    
    if not models:
        print("❌ No models found")
        return
    
    # Predict
    all_probs = []
    for path in tqdm(test_paths, desc="Predicting"):
        img = Image.open(path).convert("RGB")
        probs_list = [tta_predict(m, img, DEVICE) for m in models]
        avg_probs = torch.stack(probs_list).mean(dim=0)
        all_probs.append(avg_probs.cpu().numpy())
    
    all_probs = np.array(all_probs)
    preds = all_probs.argmax(axis=1)
    confidences = all_probs.max(axis=1)
    
    print(f"\n  Confidence: mean={confidences.mean():.3f}, min={confidences.min():.3f}, max={confidences.max():.3f}")
    
    # Save
    sub_df = pd.DataFrame({
        "path": [p.name for p in test_paths],
        "class_idx": preds,
        "class_name": [idx_to_cls[i] for i in preds],
        "confidence": confidences,
    }).sort_values("path")
    
    sub_df[["path", "class_idx"]].to_csv(OUTPUT_DIR / "submission.csv", index=False)
    sub_df.to_csv(OUTPUT_DIR / "submission_full.csv", index=False)
    
    print(f"\n✅ Saved to {OUTPUT_DIR / 'submission.csv'}")
    
    # Distribution
    print(f"\nPREDICTION DISTRIBUTION:")
    for idx, cnt in pd.Series(preds).value_counts().sort_index().items():
        print(f"  {idx_to_cls[idx]}: {cnt} ({100*cnt/len(preds):.1f}%)")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "="*70)
    print("DINOv2 SSL FINE-TUNING PIPELINE")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*70)
    
    # Phase 1: SSL fine-tuning
    train_ssl()
    
    # Phase 2: Supervised fine-tuning
    train_supervised()
    
    # Phase 3: Submission
    generate_submission()
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
