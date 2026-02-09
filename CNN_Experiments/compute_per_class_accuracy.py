"""
Compute per-class validation accuracy averaged across folds for CNN models.
Uses the class mapping from the checkpoint.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
import timm

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path("/Users/marwasulaiman/Documents/BDMA/CS - 3rd Semester/ML/Birds_Image_Classification")

TRAIN_IMG_DIR = PROJECT_ROOT / "bdma-07-competition-2026/BDMA7_project_files/train_images"
VAL_IMG_DIR = PROJECT_ROOT / "bdma-07-competition-2026/BDMA7_project_files/val_images"

CONVNEXT_DIR = PROJECT_ROOT / "CNN_Experiments/models/convnexttiny/img384_5fold_tl-last2blocks_head-mlp_ep30_tta5-mm_20260201_1405"
EFFICIENTNET_DIR = PROJECT_ROOT / "CNN_Experiments/models/efficientnetv2s/img384_5fold_tl-last1blocks_head-mlp_ep10_tta5-mm_20260128_0107"

OUTPUT_FILE = PROJECT_ROOT / "CNN_Experiments/per_class_accuracy_results.json"

IMG_SIZE = 384
N_FOLDS = 5


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


class BirdDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def load_data(class_to_idx: dict):
    """Load data using the class mapping from checkpoint."""
    image_paths = []
    labels = []
    
    for data_dir in [TRAIN_IMG_DIR, VAL_IMG_DIR]:
        if not data_dir.exists():
            continue
        for class_dir in sorted(data_dir.iterdir()):
            if not class_dir.is_dir() or class_dir.name not in class_to_idx:
                continue
            for img_path in class_dir.glob("*.jpg"):
                image_paths.append(str(img_path))
                labels.append(class_to_idx[class_dir.name])
    
    return np.array(image_paths), np.array(labels)


def build_model(model_name: str, state_dict: dict, num_classes: int = 20):
    """Build model matching the checkpoint architecture."""
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    
    if "head.fc.0.weight" in state_dict:
        # ConvNeXt with MLP head
        model.head.fc = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 20),
        )
    elif "classifier.0.weight" in state_dict:
        # EfficientNet with MLP head
        in_features = state_dict["classifier.0.weight"].shape[1]
        hidden_dim = state_dict["classifier.0.weight"].shape[0]
        model.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )
    
    model.load_state_dict(state_dict, strict=True)
    return model


def evaluate_per_class(model, dataloader, idx_to_class, device):
    model.eval()
    
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            
            for pred, label in zip(preds, labels):
                label_idx = label.item()
                class_name = idx_to_class[label_idx]
                class_total[class_name] += 1
                if pred == label:
                    class_correct[class_name] += 1
    
    per_class_acc = {}
    for class_name in class_total:
        if class_total[class_name] > 0:
            per_class_acc[class_name] = 100 * class_correct[class_name] / class_total[class_name]
    
    return per_class_acc


def compute_model_per_class_accuracy(model_dir: Path, model_name_display: str):
    print(f"\n{'='*60}")
    print(f"Processing: {model_name_display}")
    print(f"{'='*60}")
    
    config_path = model_dir / "experiment_config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    model_name = config["model_name"]
    
    # Get class mapping from first checkpoint
    first_ckpt = torch.load(model_dir / "fold1.pth", map_location="cpu")
    class_to_idx = first_ckpt["class_mapping"]  # {class_name: idx}
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    
    print(f"  Classes: {num_classes}")
    
    # Load data with checkpoint's class mapping
    image_paths, labels = load_data(class_to_idx)
    print(f"  Total images: {len(image_paths)}")
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_class_accs = defaultdict(list)
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels)):
        fold_num = fold_idx + 1
        checkpoint_path = model_dir / f"fold{fold_num}.pth"
        
        if not checkpoint_path.exists():
            continue
        
        print(f"  Fold {fold_num}: Loading...")
        
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"]
        
        model = build_model(model_name, state_dict, num_classes)
        model = model.to(DEVICE)
        model.eval()
        
        val_paths = image_paths[val_idx]
        val_labels = labels[val_idx]
        
        val_dataset = BirdDataset(val_paths, val_labels, transform)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        per_class_acc = evaluate_per_class(model, val_loader, idx_to_class, DEVICE)
        
        for class_name, acc in per_class_acc.items():
            fold_class_accs[class_name].append(acc)
        
        overall = np.mean(list(per_class_acc.values()))
        print(f"  Fold {fold_num}: {overall:.2f}%")
    
    # Compute mean and std
    results = {}
    for class_name in sorted(class_to_idx.keys()):
        accs = fold_class_accs.get(class_name, [])
        if accs:
            results[class_name] = {
                "mean": round(np.mean(accs), 2),
                "std": round(np.std(accs), 2),
            }
    
    return results


def main():
    print("\n" + "="*70)
    print("PER-CLASS ACCURACY COMPUTATION")
    print("="*70)
    print(f"Device: {DEVICE}")
    
    all_results = {}
    
    if CONVNEXT_DIR.exists():
        all_results["ConvNeXt-Tiny"] = compute_model_per_class_accuracy(CONVNEXT_DIR, "ConvNeXt-Tiny")
    
    if EFFICIENTNET_DIR.exists():
        all_results["EfficientNetV2-S"] = compute_model_per_class_accuracy(EFFICIENTNET_DIR, "EfficientNetV2-S")
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Saved to: {OUTPUT_FILE}")
    
    # Summary
    print("\n" + "="*80)
    print("RESULTS: Mean ± Std (%) across folds")
    print("="*80)
    
    all_classes = set()
    for model_results in all_results.values():
        all_classes.update(model_results.keys())
    
    print(f"\n{'Class':<30} {'ConvNeXt-Tiny':>20} {'EfficientNetV2-S':>20}")
    print("-" * 70)
    
    for class_name in sorted(all_classes):
        conv = all_results.get("ConvNeXt-Tiny", {}).get(class_name, {})
        eff = all_results.get("EfficientNetV2-S", {}).get(class_name, {})
        
        conv_str = f"{conv.get('mean', 0):.1f}±{conv.get('std', 0):.1f}" if conv else "N/A"
        eff_str = f"{eff.get('mean', 0):.1f}±{eff.get('std', 0):.1f}" if eff else "N/A"
        
        print(f"{class_name:<30} {conv_str:>20} {eff_str:>20}")


if __name__ == "__main__":
    main()
