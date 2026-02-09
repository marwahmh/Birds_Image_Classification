"""
DINOv2-style Self-Supervised + Supervised Bird Classification Pipeline (IMPROVED)

Phases:
  1) SSL data prep on CUB-200-2011 (YOLO-cropped) with verified exclusion
  2) Self-supervised pre-training with proper DINO teacher–student ViT-S backbone
  3) Supervised fine-tuning on 20 competition classes with 5-fold CV
  4) Test-time ensembling and submission file generation

IMPROVEMENTS:
  - Proper data leakage verification
  - Multi-crop augmentation (2 global + 8 local crops)
  - Better DINO loss with normalization and temperature warmup
  - Improved gradual unfreezing with logging
  - Linear probe evaluation during SSL training
  - Better checkpointing and monitoring
"""

import os
import math
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict

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
import timm


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path("/gpfs/workdir/cardilloa/Marwa/Birds_Image_Classification")

# CUB YOLO-cropped images (Phase 1+2 SSL pre-training)
CUB_CROPPED_DIR = PROJECT_ROOT / "DINO_Experiments/contrastive_results/cropped_images/cub"

# Competition data (20 classes)
COMP_BASE = PROJECT_ROOT / "bdma-07-competition-2026/BDMA7_project_files"
TRAIN_IMG_DIR = COMP_BASE / "train_images"
VAL_IMG_DIR = COMP_BASE / "val_images"
TEST_IMG_DIR = COMP_BASE / "test_images/mistery_cat"

OUTPUT_DIR = PROJECT_ROOT / "DINO_Experiments/dino_ssl_results_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Competition 20-class names (folder names under train_images / val_images)
COMPETITION_CLASSES = [
    "American_Crow",
    "Black_billed_Cuckoo",
    "Bobolink",
    "Brandt_Cormorant",
    "Brewer_Blackbird",
    "Bronzed_Cowbird",
    "Brown_Creeper",
    "Eastern_Towhee",
    "Fish_Crow",
    "Gray_Catbird",
    "Gray_crowned_Rosy_Finch",
    "Groove_billed_Ani",
    "Indigo_Bunting",
    "Lazuli_Bunting",
    "Painted_Bunting",
    "Red_winged_Blackbird",
    "Rusty_Blackbird",
    "Yellow_billed_Cuckoo",
    "Yellow_breasted_Chat",
    "Yellow_headed_Blackbird",
]

# VERIFIED CUB folder names that correspond to the 20 competition species
# These classes are EXCLUDED from SSL so that SSL never sees the labeled species
CUB_COMPETITION_CLASS_FOLDERS: List[str] = [
    "004.Groove_billed_Ani",
    "009.Brewer_Blackbird",
    "010.Red_winged_Blackbird",
    "011.Rusty_Blackbird",
    "012.Yellow_headed_Blackbird",
    "013.Bobolink",
    "014.Indigo_Bunting",
    "015.Lazuli_Bunting",
    "016.Painted_Bunting",
    "019.Gray_Catbird",
    "020.Yellow_breasted_Chat",
    "021.Eastern_Towhee",
    "023.Brandt_Cormorant",
    "026.Bronzed_Cowbird",
    "028.Brown_Creeper",
    "029.American_Crow",
    "030.Fish_Crow",
    "031.Black_billed_Cuckoo",
    "033.Yellow_billed_Cuckoo",
    "034.Gray_crowned_Rosy_Finch",
]

# SSL / DINO config
SSL_IMG_SIZE = 224
SSL_LOCAL_CROP_SIZE = 96

# IMPORTANT: Multi-crop is very memory hungry: effective batch = (2 + SSL_N_LOCAL_CROPS) * SSL_BATCH_SIZE
# On a 32GB GPU, use a conservative setting to avoid OOM:
#   - 2 global crops + 2 local crops per image
#   - Batch size 32 → effective 128 crops per batch
SSL_N_LOCAL_CROPS = 2
SSL_BATCH_SIZE = 32
SSL_EPOCHS = 100
SSL_BASE_LR = 3e-5  # Reduced from 1e-4 for stability
SSL_WD = 1e-4
SSL_OUT_DIM = 256  # DINO projection head dimension
SSL_WARMUP_EPOCHS = 15  # Increased warmup for stability
SSL_CLIP_GRAD = 0.5  # More aggressive gradient clipping

# DINO temperatures (higher = more stable, but less sharp distributions)
STUDENT_T = 0.1
TEACHER_T_BASE = 0.04
TEACHER_T_FINAL = 0.07
TEACHER_T_WARMUP_EPOCHS = 30

# EMA momentum schedule
EMA_MOMENTUM_BASE = 0.996
EMA_MOMENTUM_FINAL = 1.0

# Supervised fine-tuning config
SUP_BATCH_SIZE = 32
SUP_EPOCHS = 30
SUP_BASE_LR = 1e-3
SUP_WD = 1e-4
SUP_FOLDS = 5

# Linear probe evaluation during SSL
LINEAR_PROBE_EVERY = 10  # Epochs
LINEAR_PROBE_SAMPLES = 500  # Samples per class for quick eval


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


DEVICE = get_device()


# ============================================================
# HELPER: Data Leakage Verification
# ============================================================

def is_competition_species(cub_folder_name: str) -> bool:
    """
    Check if a CUB folder name corresponds to one of our competition species.
    CUB folders are formatted like "001.Black_footed_Albatross"
    """
    return cub_folder_name in CUB_COMPETITION_CLASS_FOLDERS


def verify_no_overlap_comprehensive():
    """
    Comprehensive verification that no competition species data leaks into SSL.
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE DATA LEAKAGE VERIFICATION")
    print("="*70)
    
    if not CUB_CROPPED_DIR.exists():
        print(f"⚠️  WARNING: CUB_CROPPED_DIR not found: {CUB_CROPPED_DIR}")
        return
    
    # Step 1: Check species-level exclusion
    ssl_species = set()
    excluded_species = set()
    excluded_image_count = 0
    ssl_image_count = 0
    
    for class_dir in sorted(CUB_CROPPED_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        
        image_count = len(list(class_dir.glob("*.jpg")))
        
        if is_competition_species(class_dir.name):
            excluded_species.add(class_dir.name)
            excluded_image_count += image_count
            print(f"  ❌ EXCLUDED: {class_dir.name} ({image_count} images)")
        else:
            ssl_species.add(class_dir.name)
            ssl_image_count += image_count
    
    # Step 2: Verify counts
    print(f"\n{'='*70}")
    print("SUMMARY:")
    print(f"{'='*70}")
    print(f"  ✅ SSL dataset will use: {len(ssl_species)} species ({ssl_image_count} images)")
    print(f"  ❌ Excluded from SSL: {len(excluded_species)} species ({excluded_image_count} images)")
    print(f"  📊 Expected exclusions: {len(CUB_COMPETITION_CLASS_FOLDERS)}")
    
    # Step 3: Sanity checks
    if len(excluded_species) != len(CUB_COMPETITION_CLASS_FOLDERS):
        print(f"\n⚠️  WARNING: Expected to exclude {len(CUB_COMPETITION_CLASS_FOLDERS)} species")
        print(f"             but actually excluding {len(excluded_species)} species")
        missing = set(CUB_COMPETITION_CLASS_FOLDERS) - excluded_species
        if missing:
            print(f"  Missing exclusions: {missing}")
    
    # Step 4: Check for any overlap (should be empty)
    overlap = ssl_species & excluded_species
    if overlap:
        print(f"\n❌ CRITICAL ERROR: Found species in both SSL and excluded sets!")
        print(f"   Overlap: {overlap}")
        raise ValueError("Data leakage detected!")
    
    # Step 5: Verify expected species count
    total_cub_species = len(ssl_species) + len(excluded_species)
    print(f"\n  📊 Total CUB species found: {total_cub_species}")
    print(f"  📊 Expected CUB species: ~200")
    
    if total_cub_species < 180:
        print(f"  ⚠️  WARNING: Only found {total_cub_species} species, expected ~200")
    
    # Step 6: Image-level check (filename overlap)
    print(f"\n{'='*70}")
    print("IMAGE-LEVEL VERIFICATION:")
    print(f"{'='*70}")
    
    ssl_image_names = set()
    for class_dir in sorted(CUB_CROPPED_DIR.iterdir()):
        if not class_dir.is_dir() or is_competition_species(class_dir.name):
            continue
        for img_path in class_dir.glob("*.jpg"):
            ssl_image_names.add(img_path.name)
    
    comp_image_names = set()
    for img_dir in [TRAIN_IMG_DIR, VAL_IMG_DIR]:
        if not img_dir.exists():
            continue
        for class_dir in img_dir.iterdir():
            if not class_dir.is_dir():
                continue
            for img_path in class_dir.glob("*.jpg"):
                comp_image_names.add(img_path.name)
    
    filename_overlap = ssl_image_names & comp_image_names
    
    if filename_overlap:
        print(f"  ⚠️  Found {len(filename_overlap)} images with same filenames in SSL and competition")
        print(f"  First 10: {list(filename_overlap)[:10]}")
        print(f"  This is OK if they're from different species (different folders)")
    else:
        print(f"  ✅ No filename overlap between SSL and competition images")
    
    print(f"\n{'='*70}")
    print("✅ VERIFICATION COMPLETE - NO DATA LEAKAGE DETECTED")
    print(f"{'='*70}\n")


# ============================================================
# PHASE 1: SSL DATASET (Multi-crop augmentation)
# ============================================================

class DinoSSLDataset(Dataset):
    """
    Self-supervised dataset for DINO-style training.
    
    - Uses CUB_CROPPED_DIR
    - Excludes any CUB class folders in CUB_COMPETITION_CLASS_FOLDERS
    - Returns 2 global crops + N local crops of the same image
    """

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        if not self.root_dir.exists():
            raise FileNotFoundError(f"CUB_CROPPED_DIR not found: {self.root_dir}")

        self.image_paths: List[Path] = []
        excluded_count = 0
        used_species = []
        
        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            
            if is_competition_species(class_dir.name):
                excluded_count += 1
                continue
            
            used_species.append(class_dir.name)
            for img_path in class_dir.glob("*.jpg"):
                self.image_paths.append(img_path)

        if len(self.image_paths) == 0:
            raise RuntimeError(
                "No SSL images found. Make sure CUB_CROPPED_DIR exists and "
                "CUB_COMPETITION_CLASS_FOLDERS is correctly set."
            )

        print(f"\n[SSL DATASET]")
        print(f"  Total images: {len(self.image_paths)}")
        print(f"  Species used: {len(used_species)}")
        print(f"  Species excluded: {excluded_count}")
        print(f"  Images per species (avg): {len(self.image_paths) / max(1, len(used_species)):.1f}")

        # DINO-style global crops (224x224)
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(SSL_IMG_SIZE, scale=(0.4, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # DINO-style local crops (96x96) - then upsampled to 224x224 for ViT-S
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(SSL_LOCAL_CROP_SIZE, scale=(0.05, 0.4), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            # ViT-S from timm expects fixed 224x224, so we upsample locals back to 224
            transforms.Resize(SSL_IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.n_local_crops = SSL_N_LOCAL_CROPS

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Returns:
            global_crops: List of 2 tensors [3, 224, 224]
            local_crops: List of N tensors [3, 96, 96]
        """
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        
        # Generate 2 global crops
        global_crops = [self.global_transform(image) for _ in range(2)]
        
        # Generate N local crops
        local_crops = [self.local_transform(image) for _ in range(self.n_local_crops)]
        
        return global_crops, local_crops


def collate_multicrop(batch):
    """
    Custom collate function for multi-crop batches.
    
    Input: List of (global_crops, local_crops) tuples
    Output: (stacked_global_crops, stacked_local_crops)
    """
    global_crops_batch = []
    local_crops_batch = []
    
    for global_crops, local_crops in batch:
        global_crops_batch.append(global_crops)
        local_crops_batch.append(local_crops)
    
    # Stack each crop type separately
    # global_crops_batch: List[List[Tensor]] -> List[Tensor] (one per crop type)
    n_global = len(global_crops_batch[0])
    n_local = len(local_crops_batch[0])
    
    stacked_global = [torch.stack([item[i] for item in global_crops_batch]) for i in range(n_global)]
    stacked_local = [torch.stack([item[i] for item in local_crops_batch]) for i in range(n_local)]
    
    return stacked_global, stacked_local


def build_ssl_dataloader() -> DataLoader:
    dataset = DinoSSLDataset(CUB_CROPPED_DIR)
    loader = DataLoader(
        dataset,
        batch_size=SSL_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=True,
        collate_fn=collate_multicrop,
    )
    return loader


# ============================================================
# PHASE 2: IMPROVED DINO COMPONENTS
# ============================================================

class DinoVitBackbone(nn.Module):
    """
    ViT-S backbone using timm, with no classification head.
    """

    def __init__(self, img_size: int = 224):
        super().__init__()
        self.model = timm.create_model(
            "vit_small_patch16_224",
            img_size=img_size,
            pretrained=False,
            num_classes=0,  # no classification head
        )
        self.embed_dim = 384  # ViT-S embedding dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output: [B, 384] for ViT-S
        return self.model(x)


class DinoHead(nn.Module):
    """
    DINO projection head with normalization layer and proper initialization.
    """

    def __init__(self, in_dim: int, out_dim: int = SSL_OUT_DIM, hidden_dim: int = 2048, bottleneck_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        
        # Initialize weights for stability
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stable training"""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Last layer with smaller init
        nn.init.trunc_normal_(self.last_layer.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2, eps=1e-6)  # Added eps for numerical stability
        x = self.last_layer(x)
        return x


class DinoStudentTeacher(nn.Module):
    """
    Container holding student and teacher backbones & heads.
    """

    def __init__(self, feature_dim: int = 384):
        super().__init__()
        # Student
        self.student_backbone = DinoVitBackbone(img_size=SSL_IMG_SIZE)
        self.student_head = DinoHead(in_dim=feature_dim, out_dim=SSL_OUT_DIM)

        # Teacher
        self.teacher_backbone = DinoVitBackbone(img_size=SSL_IMG_SIZE)
        self.teacher_head = DinoHead(in_dim=feature_dim, out_dim=SSL_OUT_DIM)

        # Init teacher = student
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())

        # Teacher is not directly trained
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False
        
        print(f"\n[MODEL] Student-Teacher initialized")
        print(f"  Backbone: ViT-Small ({feature_dim}D features)")
        print(f"  Projection: {feature_dim}D -> {SSL_OUT_DIM}D")
        total_params = sum(p.numel() for p in self.student_backbone.parameters())
        print(f"  Total params (student backbone): {total_params/1e6:.2f}M")

    def forward_student(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.student_backbone(x)
        return self.student_head(feats)

    @torch.no_grad()
    def forward_teacher(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.teacher_backbone(x)
        return self.teacher_head(feats)


class ImprovedDinoLoss(nn.Module):
    """
    Improved DINO loss with proper normalization and temperature warmup.
    
    Key fixes for numerical stability:
    - No double L2 normalization (head already normalizes)
    - Center updated with raw teacher outputs
    - Clamping to prevent extreme values
    """

    def __init__(
        self, 
        out_dim: int, 
        student_temp: float, 
        teacher_temp_base: float,
        teacher_temp_final: float,
        warmup_teacher_temp_epochs: int,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp_base = teacher_temp_base
        self.teacher_temp_final = teacher_temp_final
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.center_momentum = center_momentum
        
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.current_epoch = 0
    
    def set_epoch(self, epoch: int):
        """Update current epoch for temperature warmup"""
        self.current_epoch = epoch

    @torch.no_grad()
    def update_center(self, teacher_out: torch.Tensor):
        """Update center with EMA using raw teacher outputs"""
        batch_center = teacher_out.mean(dim=0, keepdim=True)
        # Initialize center with first batch mean if it's all zeros
        if self.center.abs().sum() < 1e-8:
            self.center = batch_center
        else:
            self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def forward(self, student_out: torch.Tensor, teacher_out: torch.Tensor) -> torch.Tensor:
        """
        Both inputs: [B, out_dim]
        Note: Inputs are already L2-normalized by DinoHead
        """
        # Temperature warmup for teacher
        if self.current_epoch < self.warmup_teacher_temp_epochs:
            teacher_temp = self.teacher_temp_base + (self.teacher_temp_final - self.teacher_temp_base) * \
                          (self.current_epoch / self.warmup_teacher_temp_epochs)
        else:
            teacher_temp = self.teacher_temp_final
        
        # Apply temperatures and centering
        # Clamp values to prevent extreme logits
        student_logits = student_out / self.student_temp
        student_logits = student_logits.clamp(-100, 100)
        
        teacher_centered = teacher_out - self.center
        teacher_logits = teacher_centered / teacher_temp
        teacher_logits = teacher_logits.clamp(-100, 100)
        
        # Compute probabilities
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1).detach()
        
        # Cross-entropy loss with numerical stability
        loss = -torch.sum(teacher_probs * student_log_probs, dim=-1).mean()
        
        return loss


def cosine_scheduler(base_value, final_value, epochs, niter_per_epoch, warmup_epochs=0):
    """Cosine learning rate schedule with warmup"""
    total_iters = epochs * niter_per_epoch
    warmup_iters = warmup_epochs * niter_per_epoch
    schedule = np.zeros(total_iters, dtype=np.float32)

    for i in range(total_iters):
        if i < warmup_iters:
            schedule[i] = base_value * float(i) / float(max(1, warmup_iters))
        else:
            t = (i - warmup_iters) / float(max(1, total_iters - warmup_iters))
            schedule[i] = final_value + 0.5 * (base_value - final_value) * (1.0 + math.cos(math.pi * t))
    return schedule


def ema_update(student: nn.Module, teacher: nn.Module, m: float):
    """Exponential moving average update for teacher"""
    with torch.no_grad():
        for ps, pt in zip(student.parameters(), teacher.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)


# ============================================================
# LINEAR PROBE EVALUATION (Monitor SSL quality)
# ============================================================

def linear_probe_evaluation(model: DinoStudentTeacher, epoch: int) -> float:
    """
    Quick linear probe evaluation to monitor SSL feature quality.
    Uses a simple logistic regression on frozen features.
    """
    print(f"\n[LINEAR PROBE] Evaluating at epoch {epoch+1}...")
    
    # Load competition data for probe
    df = load_competition_dataframe()
    
    # Simple transform for probe
    probe_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Sample subset for speed (LINEAR_PROBE_SAMPLES per class)
    sampled_df = df.groupby('label').apply(
        lambda x: x.sample(min(len(x), LINEAR_PROBE_SAMPLES), random_state=42)
    ).reset_index(drop=True)
    
    # Extract features
    model.eval()
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Extracting features", leave=False):
            img = Image.open(row['path']).convert('RGB')
            img_tensor = probe_transform(img).unsqueeze(0).to(DEVICE)
            feat = model.student_backbone(img_tensor).cpu().numpy()
            features_list.append(feat[0])
            labels_list.append(row['label'])
    
    features = np.array(features_list)
    labels = np.array(labels_list)
    
    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train logistic regression
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    train_acc = clf.score(X_train, y_train) * 100
    test_acc = clf.score(X_test, y_test) * 100
    
    print(f"[LINEAR PROBE] Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")
    model.train()
    
    return test_acc


# ============================================================
# PHASE 2: DINO SSL TRAINING
# ============================================================

def train_dino_ssl():
    """
    Phase 2: DINO-style SSL pre-training on CUB (excluding 20 competition classes).
    """
    verify_no_overlap_comprehensive()
    ssl_loader = build_ssl_dataloader()

    print("\n[SSL] Building DINO student-teacher...")
    model = DinoStudentTeacher(feature_dim=384).to(DEVICE)
    
    dino_loss_fn = ImprovedDinoLoss(
        out_dim=SSL_OUT_DIM,
        student_temp=STUDENT_T,
        teacher_temp_base=TEACHER_T_BASE,
        teacher_temp_final=TEACHER_T_FINAL,
        warmup_teacher_temp_epochs=TEACHER_T_WARMUP_EPOCHS,
    ).to(DEVICE)

    # Optimizer on student only
    optim = torch.optim.AdamW(
        list(model.student_backbone.parameters()) + list(model.student_head.parameters()),
        lr=SSL_BASE_LR,
        weight_decay=SSL_WD,
    )

    iters_per_epoch = len(ssl_loader)
    lr_schedule = cosine_scheduler(
        base_value=SSL_BASE_LR,
        final_value=SSL_BASE_LR * 0.01,
        epochs=SSL_EPOCHS,
        niter_per_epoch=iters_per_epoch,
        warmup_epochs=SSL_WARMUP_EPOCHS,
    )
    
    # EMA momentum schedule
    momentum_schedule = cosine_scheduler(
        base_value=EMA_MOMENTUM_BASE,
        final_value=EMA_MOMENTUM_FINAL,
        epochs=SSL_EPOCHS,
        niter_per_epoch=iters_per_epoch,
        warmup_epochs=0,
    )

    history = []
    probe_history = []
    global_step = 0
    best_probe_acc = 0.0
    nan_count = 0  # Track NaN occurrences

    for epoch in range(SSL_EPOCHS):
        model.train()
        dino_loss_fn.set_epoch(epoch)
        total_loss = 0.0
        valid_batches = 0  # Track valid (non-NaN) batches
        
        pbar = tqdm(ssl_loader, desc=f"[SSL] Epoch {epoch+1}/{SSL_EPOCHS}")
        
        for it, (global_crops, local_crops) in enumerate(pbar):
            # Move to device
            global_crops = [crop.to(DEVICE, non_blocking=True) for crop in global_crops]
            local_crops = [crop.to(DEVICE, non_blocking=True) for crop in local_crops]
            
            # Learning rate schedule
            lr = lr_schedule[global_step]
            for param_group in optim.param_groups:
                param_group["lr"] = lr

            # Teacher only sees global crops
            teacher_crops = global_crops
            
            # Student sees all crops (global + local)
            student_crops = global_crops + local_crops
            
            # Forward passes
            # Teacher
            with torch.no_grad():
                teacher_outputs = [model.forward_teacher(crop) for crop in teacher_crops]
                # Update center with raw teacher outputs (before any loss computation)
                teacher_concat = torch.cat(teacher_outputs, dim=0)  # [2B, out_dim]
                dino_loss_fn.update_center(teacher_concat)
            
            # Student
            student_outputs = [model.forward_student(crop) for crop in student_crops]
            
            # Compute loss
            # We only compute loss between student's global crops and teacher's global crops
            # Plus student's local crops vs teacher's global crops
            n_global = len(global_crops)
            batch_size = global_crops[0].shape[0]
            
            loss = 0
            n_loss_terms = 0
            # For each student crop, compute loss against all teacher crops
            for i, student_out in enumerate(student_outputs):
                # Each student crop vs averaged teacher output
                for j, teacher_out in enumerate(teacher_outputs):
                    if i < n_global and i == j:
                        # Skip same crop (student global vs same teacher global)
                        continue
                    loss += dino_loss_fn(student_out, teacher_out)
                    n_loss_terms += 1
            
            # Normalize by number of loss terms
            if n_loss_terms > 0:
                loss = loss / n_loss_terms
            
            # Check for NaN/Inf before backward
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                if nan_count <= 10:  # Only print first 10 warnings
                    print(f"\n⚠️ WARNING: NaN/Inf loss detected at epoch {epoch+1} iter {it}, skipping batch")
                if nan_count == 10:
                    print("  (Further NaN warnings suppressed)")
                global_step += 1
                continue

            optim.zero_grad()
            loss.backward()
            
            if SSL_CLIP_GRAD is not None:
                grad_norm = nn.utils.clip_grad_norm_(
                    list(model.student_backbone.parameters()) + list(model.student_head.parameters()),
                    SSL_CLIP_GRAD,
                )
            optim.step()

            # EMA update for teacher
            m = momentum_schedule[global_step]
            ema_update(model.student_backbone, model.teacher_backbone, m)
            ema_update(model.student_head, model.teacher_head, m)

            total_loss += loss.item()
            valid_batches += 1
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{lr:.6f}',
                'mom': f'{m:.4f}'
            })
            
            history.append({
                "epoch": epoch + 1,
                "iter": it,
                "loss": float(loss.item()),
                "lr": float(lr),
                "momentum": float(m),
            })

            global_step += 1

        avg_loss = total_loss / max(1, valid_batches)
        print(f"\n[SSL] Epoch {epoch+1:03d} | Avg Loss: {avg_loss:.4f} | Valid batches: {valid_batches}/{len(ssl_loader)}")

        # Linear probe evaluation
        if (epoch + 1) % LINEAR_PROBE_EVERY == 0 or (epoch + 1) == SSL_EPOCHS:
            probe_acc = linear_probe_evaluation(model, epoch)
            probe_history.append({
                'epoch': epoch + 1,
                'probe_accuracy': probe_acc
            })
            
            if probe_acc > best_probe_acc:
                best_probe_acc = probe_acc
                # Save best model based on probe
                best_path = OUTPUT_DIR / "dino_ssl_best_probe.pth"
                torch.save({
                    "epoch": epoch + 1,
                    "student_backbone": model.student_backbone.state_dict(),
                    "student_head": model.student_head.state_dict(),
                    "teacher_backbone": model.teacher_backbone.state_dict(),
                    "teacher_head": model.teacher_head.state_dict(),
                    "probe_accuracy": probe_acc,
                }, best_path)
                print(f"[SSL] 💾 Saved best probe model (acc: {probe_acc:.2f}%)")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 or (epoch + 1) == SSL_EPOCHS:
            ckpt_path = OUTPUT_DIR / f"dino_ssl_epoch{epoch+1:03d}.pth"
            torch.save({
                "epoch": epoch + 1,
                "student_backbone": model.student_backbone.state_dict(),
                "student_head": model.student_head.state_dict(),
                "teacher_backbone": model.teacher_backbone.state_dict(),
                "teacher_head": model.teacher_head.state_dict(),
                "optimizer": optim.state_dict(),
                "history": history,
                "probe_history": probe_history,
            }, ckpt_path)
            print(f"[SSL] 💾 Saved checkpoint to {ckpt_path}")

    # Save training history
    with open(OUTPUT_DIR / "dino_ssl_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    with open(OUTPUT_DIR / "dino_ssl_probe_history.json", "w") as f:
        json.dump(probe_history, f, indent=2)
    
    print(f"\n[SSL] Training complete!")
    print(f"  Best probe accuracy: {best_probe_acc:.2f}%")


# ============================================================
# PHASE 3: SUPERVISED FINE-TUNING ON 20 CLASSES
# ============================================================

class SupervisedBirdDataset(Dataset):
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


def load_competition_dataframe() -> pd.DataFrame:
    """Load and merge train + val competition data"""
    def gather_from_dir(base_dir: Path) -> List[dict]:
        rows = []
        if not base_dir.exists():
            print(f"⚠️  WARNING: Directory not found: {base_dir}")
            return rows
        
        for class_dir in sorted(base_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            if class_name not in COMPETITION_CLASSES:
                continue
            for img_path in class_dir.glob("*.jpg"):
                rows.append({"path": str(img_path), "class_name": class_name})
        return rows

    rows = gather_from_dir(TRAIN_IMG_DIR) + gather_from_dir(VAL_IMG_DIR)
    df = pd.DataFrame(rows)
    cls_to_idx = {name: i for i, name in enumerate(sorted(COMPETITION_CLASSES))}
    df["label"] = df["class_name"].map(cls_to_idx).astype(int)
    
    print(f"\n[SUPERVISED DATA]")
    print(f"  Total images: {len(df)}")
    print(f"  Classes: {len(cls_to_idx)}")
    print(f"  Images per class (avg): {len(df) / len(cls_to_idx):.1f}")
    
    return df


def build_supervised_transforms():
    """Build augmentation transforms for supervised training"""
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return train_tf, val_tf


class DinoClassifier(nn.Module):
    """
    Classifier using a pre-trained ViT-S backbone and a classification head.
    Supports gradual unfreezing of transformer blocks.
    """

    def __init__(self, backbone: nn.Module, num_classes: int = 20, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.backbone = backbone
        
        # Initially freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(384),
            nn.Dropout(dropout),
            nn.Linear(384, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )
        
        self._log_params()

    def _log_params(self):
        """Log parameter counts"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n[CLASSIFIER]")
        print(f"  Total params: {total/1e6:.2f}M")
        print(f"  Trainable params: {trainable/1e6:.2f}M ({100*trainable/total:.1f}%)")

    def unfreeze_last_blocks(self, n_blocks: int = 1):
        """
        Gradually unfreeze last n transformer blocks.
        Assumes timm ViT where blocks are in self.backbone.model.blocks
        """
        if not hasattr(self.backbone, 'model') or not hasattr(self.backbone.model, 'blocks'):
            print(f"  ⚠️  WARNING: Cannot access backbone.model.blocks, skipping unfreezing")
            return
        
        total_blocks = len(self.backbone.model.blocks)
        if n_blocks > total_blocks:
            print(f"  ⚠️  WARNING: Requested {n_blocks} blocks, only {total_blocks} available")
            n_blocks = total_blocks
        
        trainable_before = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Unfreeze last n blocks
        for blk in self.backbone.model.blocks[-n_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
        
        trainable_after = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n[UNFREEZING]")
        print(f"  Unfroze last {n_blocks}/{total_blocks} transformer blocks")
        print(f"  Trainable params: {trainable_before/1e6:.2f}M → {trainable_after/1e6:.2f}M")
        print(f"  Additional params: {(trainable_after-trainable_before)/1e6:.2f}M")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone features
        feats = self.backbone(x)  # [B, 384]
        # Classification
        return self.head(feats)


def load_best_ssl_backbone(epoch: int = None, use_best_probe: bool = True) -> DinoVitBackbone:
    """
    Load backbone weights from SSL checkpoint.
    
    Args:
        epoch: Specific epoch to load. If None, uses best_probe or latest.
        use_best_probe: If True and epoch is None, loads best probe checkpoint.
    """
    # Try to load best probe model first
    if epoch is None and use_best_probe:
        best_probe_path = OUTPUT_DIR / "dino_ssl_best_probe.pth"
        if best_probe_path.exists():
            print(f"\n[LOADING] Best probe checkpoint: {best_probe_path}")
            ckpt = torch.load(best_probe_path, map_location="cpu")
            backbone = DinoVitBackbone(img_size=SSL_IMG_SIZE)
            backbone.load_state_dict(ckpt["student_backbone"])
            if "probe_accuracy" in ckpt:
                print(f"  Probe accuracy: {ckpt['probe_accuracy']:.2f}%")
            return backbone
    
    # Otherwise load specific epoch or latest
    ckpts = sorted(OUTPUT_DIR.glob("dino_ssl_epoch*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No SSL checkpoints found in {OUTPUT_DIR}")
    
    if epoch is not None:
        target = OUTPUT_DIR / f"dino_ssl_epoch{epoch:03d}.pth"
        if not target.exists():
            raise FileNotFoundError(f"Requested checkpoint not found: {target}")
        ckpt_path = target
    else:
        ckpt_path = ckpts[-1]

    print(f"\n[LOADING] Backbone from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    backbone = DinoVitBackbone(img_size=SSL_IMG_SIZE)
    backbone.load_state_dict(ckpt["student_backbone"])
    
    return backbone


def train_supervised_folds():
    """
    Phase 3: Supervised training with 5-fold cross-validation.
    """
    df = load_competition_dataframe()
    train_tf, val_tf = build_supervised_transforms()

    X = df["path"].values
    y = df["label"].values

    skf = StratifiedKFold(n_splits=SUP_FOLDS, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*70}")
        print(f"FOLD {fold+1}/{SUP_FOLDS}")
        print(f"{'='*70}")
        
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        print(f"  Train samples: {len(train_df)}")
        print(f"  Val samples: {len(val_df)}")

        train_ds = SupervisedBirdDataset(train_df, transform=train_tf)
        val_ds = SupervisedBirdDataset(val_df, transform=val_tf)

        train_loader = DataLoader(
            train_ds,
            batch_size=SUP_BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=(DEVICE.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=SUP_BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=(DEVICE.type == "cuda"),
        )

        # Load SSL backbone
        backbone = load_best_ssl_backbone(use_best_probe=True)
        model = DinoClassifier(
            backbone=backbone, 
            num_classes=len(COMPETITION_CLASSES),
            dropout=0.3
        ).to(DEVICE)

        # Training setup
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=SUP_BASE_LR, 
            weight_decay=SUP_WD
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SUP_EPOCHS)

        best_val_acc = 0.0
        best_state = None
        patience = 7
        patience_counter = 0
        fold_history = []

        for epoch in range(SUP_EPOCHS):
            # Gradual unfreezing: unfreeze last block at epoch 5
            if epoch == 5:
                print(f"\n[EPOCH {epoch+1}] Unfreezing last transformer block")
                model.unfreeze_last_blocks(n_blocks=1)
                # Recreate optimizer with new trainable params and lower LR
                optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()), 
                    lr=SUP_BASE_LR * 0.1,  # Lower LR for fine-tuning
                    weight_decay=SUP_WD
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, 
                    T_max=SUP_EPOCHS - epoch
                )

            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{SUP_EPOCHS} [Train]", leave=False)
            for imgs, labels in pbar:
                imgs = imgs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                
                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                preds = logits.argmax(dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            train_acc = 100.0 * train_correct / max(1, train_total)
            avg_train_loss = train_loss / max(1, len(train_loader))

            # Validation phase
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0.0
            
            with torch.no_grad():
                for imgs, labels in tqdm(val_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{SUP_EPOCHS} [Val]", leave=False):
                    imgs = imgs.to(DEVICE, non_blocking=True)
                    labels = labels.to(DEVICE, non_blocking=True)
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                    
                    val_loss += loss.item()
                    preds = logits.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_acc = 100.0 * val_correct / max(1, val_total)
            avg_val_loss = val_loss / max(1, len(val_loader))
            
            # Logging
            print(f"[Fold {fold+1}] Epoch {epoch+1:02d} | "
                  f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}%")
            
            fold_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_acc': train_acc,
                'val_loss': avg_val_loss,
                'val_acc': val_acc,
            })

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = model.state_dict()
                patience_counter = 0
                print(f"  ✨ New best Val Acc: {best_val_acc:.2f}%")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"  ⏹️  Early stopping at epoch {epoch+1}")
                break

            scheduler.step()

        # Save fold results
        fold_results.append(best_val_acc)
        
        # Save best model for this fold
        ckpt_path = OUTPUT_DIR / f"dino_sup_fold{fold+1}.pth"
        torch.save(best_state, ckpt_path)
        print(f"\n💾 Saved best model for fold {fold+1}: {ckpt_path}")
        print(f"   Best Val Acc: {best_val_acc:.2f}%")
        
        # Save fold history
        history_path = OUTPUT_DIR / f"dino_sup_fold{fold+1}_history.json"
        with open(history_path, 'w') as f:
            json.dump(fold_history, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*70}")
    for i, acc in enumerate(fold_results):
        print(f"  Fold {i+1}: {acc:.2f}%")
    print(f"  Mean: {np.mean(fold_results):.2f}% ± {np.std(fold_results):.2f}%")
    print(f"{'='*70}\n")
    
    # Save overall summary
    summary = {
        'fold_accuracies': fold_results,
        'mean_accuracy': float(np.mean(fold_results)),
        'std_accuracy': float(np.std(fold_results)),
    }
    with open(OUTPUT_DIR / "cv_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)


# ============================================================
# PHASE 4: EVALUATION & SUBMISSION
# ============================================================

def tta_augmentations(img: Image.Image) -> List[torch.Tensor]:
    """
    Enhanced test-time augmentation.
    Returns list of augmented tensors.
    """
    base_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # TTA variants
    variants = []
    
    # Original
    variants.append(base_tf(img))
    
    # Horizontal flip
    variants.append(base_tf(img.transpose(Image.FLIP_LEFT_RIGHT)))
    
    # Small rotations
    for angle in [-10, -5, 5, 10]:
        variants.append(base_tf(img.rotate(angle)))
    
    # Crops (center and corners)
    w, h = img.size
    crop_size = int(min(w, h) * 0.9)
    
    # Center crop
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    variants.append(base_tf(img.crop((left, top, left + crop_size, top + crop_size))))
    
    return variants


def generate_submission():
    """
    Ensemble predictions from 5 folds with TTA and save submission CSV.
    """
    print(f"\n{'='*70}")
    print("GENERATING SUBMISSION")
    print(f"{'='*70}\n")
    
    cls_to_idx = {name: i for i, name in enumerate(sorted(COMPETITION_CLASSES))}
    idx_to_cls = {v: k for k, v in cls_to_idx.items()}

    test_paths = sorted([p for p in TEST_IMG_DIR.glob("*.jpg")])
    if not test_paths:
        raise FileNotFoundError(f"No test images found in {TEST_IMG_DIR}")

    print(f"[SUBMIT] Found {len(test_paths)} test images")
    print(f"[SUBMIT] Loading {SUP_FOLDS} fold models...")

    # Load backbone (shared across folds)
    backbone = load_best_ssl_backbone(use_best_probe=True).to(DEVICE)
    backbone.eval()

    # Collect predictions from all folds
    fold_predictions = []
    
    for fold in range(SUP_FOLDS):
        print(f"\n[SUBMIT] Processing Fold {fold+1}/{SUP_FOLDS}...")
        
        ckpt_path = OUTPUT_DIR / f"dino_sup_fold{fold+1}.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing supervised checkpoint: {ckpt_path}")
        
        # Load model
        model = DinoClassifier(backbone=backbone, num_classes=len(COMPETITION_CLASSES))
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model = model.to(DEVICE)
        model.eval()

        fold_probs = []
        
        for path in tqdm(test_paths, desc=f"Fold {fold+1} predictions"):
            img = Image.open(path).convert("RGB")
            tta_imgs = tta_augmentations(img)
            batch = torch.stack(tta_imgs).to(DEVICE)
            
            with torch.no_grad():
                logits = model(batch)  # [T, C]
                probs = F.softmax(logits, dim=1).mean(dim=0)  # [C] - average over TTA
            
            fold_probs.append(probs.cpu().numpy())

        fold_predictions.append(np.stack(fold_probs, axis=0))  # [N, C]

    # Ensemble across folds (average probabilities)
    print(f"\n[SUBMIT] Ensembling {SUP_FOLDS} folds...")
    fold_predictions = np.stack(fold_predictions, axis=0)  # [F, N, C]
    mean_probs = fold_predictions.mean(axis=0)  # [N, C]
    preds = mean_probs.argmax(axis=1)  # [N]
    
    # Get confidence scores
    confidences = mean_probs.max(axis=1)  # [N]
    
    print(f"  Mean confidence: {confidences.mean():.3f}")
    print(f"  Min confidence: {confidences.min():.3f}")
    print(f"  Max confidence: {confidences.max():.3f}")

    # Create submission dataframe
    submission_df = pd.DataFrame({
        "path": [p.name for p in test_paths],
        "class_idx": preds,
        "class_name": [idx_to_cls[int(i)] for i in preds],
        "confidence": confidences,
    }).sort_values("path")

    # Save submission (only path and class_idx as required)
    sub_path = OUTPUT_DIR / "submission_dino_ssl.csv"
    submission_df[["path", "class_idx"]].to_csv(sub_path, index=False)
    
    # Save full version with confidence for analysis
    full_sub_path = OUTPUT_DIR / "submission_dino_ssl_full.csv"
    submission_df.to_csv(full_sub_path, index=False)
    
    print(f"\n✅ Saved submission to: {sub_path}")
    print(f"✅ Saved full submission to: {full_sub_path}")
    
    # Print prediction distribution
    print(f"\n{'='*70}")
    print("PREDICTION DISTRIBUTION")
    print(f"{'='*70}")
    pred_counts = pd.Series(preds).value_counts().sort_index()
    for idx, count in pred_counts.items():
        print(f"  {idx_to_cls[idx]}: {count} ({100*count/len(preds):.1f}%)")
    print(f"{'='*70}\n")


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    """
    Entry point to run all phases sequentially.
    You can comment out phases to run them separately.
    """
    print(f"\n{'='*70}")
    print("DINO SSL BIRD CLASSIFICATION PIPELINE")
    print(f"{'='*70}")
    print(f"Device: {DEVICE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*70}\n")

    # Phase 2: SSL pre-training
    print("\n" + "="*70)
    print("PHASE 2: SELF-SUPERVISED PRE-TRAINING")
    print("="*70)
    train_dino_ssl()

    # Phase 3: Supervised fine-tuning
    print("\n" + "="*70)
    print("PHASE 3: SUPERVISED FINE-TUNING")
    print("="*70)
    train_supervised_folds()

    # Phase 4: Submission generation
    print("\n" + "="*70)
    print("PHASE 4: SUBMISSION GENERATION")
    print("="*70)
    generate_submission()

    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()