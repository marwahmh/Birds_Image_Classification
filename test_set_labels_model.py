"""
Match mystery test images using deep learning features (ResNet)
Much more robust to heavy modifications than perceptual hashing
"""

import os
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors

class FeatureExtractor:
    """Extract deep features from images using pretrained ResNet"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {self.device}")
        
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=True)
        # Remove final classification layer
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.model = self.model.to(device)
        self.model.eval()
        
        # Standard ImageNet preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract(self, image_path):
        """Extract feature vector from image"""
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(img_tensor)
            
            # Flatten to 1D vector
            features = features.view(features.size(0), -1).cpu().numpy()
            return features[0]
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

def load_cub_metadata(cub_root, filter_classes=None):
    """
    Load CUB-200-2011 dataset metadata
    
    Args:
        cub_root: Path to CUB dataset
        filter_classes: List of class names to keep (None = keep all)
    """
    print("\nLoading CUB metadata...")
    
    images_file = os.path.join(cub_root, 'images.txt')
    images_df = pd.read_csv(images_file, sep=' ', header=None, names=['image_id', 'image_path'])
    
    labels_file = os.path.join(cub_root, 'image_class_labels.txt')
    labels_df = pd.read_csv(labels_file, sep=' ', header=None, names=['image_id', 'class_id'])
    
    classes_file = os.path.join(cub_root, 'classes.txt')
    classes_df = pd.read_csv(classes_file, sep=' ', header=None, names=['class_id', 'class_name'])
    
    metadata = images_df.merge(labels_df, on='image_id')
    metadata = metadata.merge(classes_df, on='class_id')
    
    # Filter to only specified classes if provided
    if filter_classes is not None:
        # Normalize class names to match CUB format (has numeric prefix)
        # CUB format: "004.Groove_billed_Ani"
        # Your format: "Groove_billed_Ani"
        
        print(f"\nFiltering to {len(filter_classes)} specified classes...")
        
        # Create filter by checking if class_name ends with any of the target classes
        def matches_filter(class_name):
            # Extract just the bird name part (after the number and dot)
            bird_name = class_name.split('.', 1)[1] if '.' in class_name else class_name
            return bird_name in filter_classes
        
        metadata = metadata[metadata['class_name'].apply(matches_filter)]
        # IMPORTANT: Reset index after filtering to avoid index misalignment
        metadata = metadata.reset_index(drop=True)
        print(f"  ✓ Filtered to {len(metadata)} images from {metadata['class_id'].nunique()} classes")
    else:
        print(f"  ✓ {len(metadata)} images from {metadata['class_id'].nunique()} classes")
    
    return metadata

def extract_cub_features(cub_root, metadata, extractor, cache_file='cub_features.npz'):
    """Extract features for all CUB images (with caching)"""
    
    # Check if cached features exist
    if os.path.exists(cache_file):
        print(f"\n✓ Loading cached features from {cache_file}")
        data = np.load(cache_file)
        return data['features'], data['image_ids']
    
    print("\nExtracting features from CUB dataset...")
    print("(This will take ~5-10 minutes, but will be cached)")
    
    images_root = os.path.join(cub_root, 'images')
    features_list = []
    image_ids = []  # Store image_ids instead of dataframe indices
    
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Extracting"):
        image_path = os.path.join(images_root, row['image_path'])
        
        if os.path.exists(image_path):
            features = extractor.extract(image_path)
            if features is not None:
                features_list.append(features)
                image_ids.append(row['image_id'])  # Store image_id
    
    features_array = np.vstack(features_list)
    image_ids = np.array(image_ids)
    
    # Cache the features
    np.savez(cache_file, features=features_array, image_ids=image_ids)
    print(f"  ✓ Cached features to {cache_file}")
    print(f"  ✓ Extracted features for {len(features_list)} images")
    
    return features_array, image_ids

def match_test_images(test_dir, cub_features, cub_metadata, image_ids, extractor, k=5, target_classes=None):
    """
    Match test images using k-nearest neighbors in feature space
    
    Args:
        k: Number of nearest neighbors to consider (default 5)
        image_ids: Array of image_ids corresponding to cub_features
        target_classes: List of target class names for index mapping
    """
    # Create mapping from class name to target index (0-19)
    class_to_idx = {}
    if target_classes is not None:
        class_to_idx = {cls: idx for idx, cls in enumerate(target_classes)}
    
    # Create mapping from image_id to metadata row for fast lookup
    metadata_by_id = {row['image_id']: row for _, row in cub_metadata.iterrows()}
    
    # Get test images
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        test_images.extend(Path(test_dir).rglob(ext))
    
    if len(test_images) == 0:
        print(f"\n❌ No images found in {test_dir}")
        return pd.DataFrame()
    
    print(f"\n✓ Found {len(test_images)} test images")
    print(f"Extracting features and finding {k} nearest neighbors...")
    
    # Build k-NN index
    knn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
    knn.fit(cub_features)
    
    results = []
    
    for test_path in tqdm(test_images, desc="Matching"):
        # Extract features for test image
        test_features = extractor.extract(str(test_path))
        
        if test_features is not None:
            # Find k nearest neighbors
            distances, indices = knn.kneighbors([test_features])
            distances = distances[0]
            indices = indices[0]
            
            # Get image_ids for nearest neighbors
            neighbor_image_ids = image_ids[indices]
            
            # Get metadata for nearest neighbors
            nearest_neighbors = [metadata_by_id[img_id] for img_id in neighbor_image_ids]
            
            # Vote by class (most common class among k neighbors)
            neighbor_classes = [n['class_id'] for n in nearest_neighbors]
            from collections import Counter
            class_votes = Counter(neighbor_classes)
            predicted_class_id = class_votes.most_common(1)[0][0]
            vote_count = class_votes[predicted_class_id]
            confidence = vote_count / k  # % of neighbors that agree
            
            # Get info for the closest match
            closest_match = nearest_neighbors[0]
            closest_distance = distances[0]
            
            # Get the class name and map to 0-19 index
            predicted_class_name_full = closest_match['class_name']
            # Extract bird name without numeric prefix
            predicted_class_name = predicted_class_name_full.split('.', 1)[1] if '.' in predicted_class_name_full else predicted_class_name_full
            
            # Map to 0-19 index
            target_class_idx = class_to_idx.get(predicted_class_name, None)
            
            # Build top k classes string
            top_classes = []
            for cls_id, cnt in class_votes.most_common(3):
                cls_info = next((n for n in nearest_neighbors if n['class_id'] == cls_id), None)
                if cls_info is not None:
                    cls_name = cls_info['class_name'].split('.', 1)[1] if '.' in cls_info['class_name'] else cls_info['class_name']
                    top_classes.append(f"{cls_name}({cnt})")
            
            results.append({
                'test_image_path': str(test_path),
                'test_image_name': test_path.name,
                'class_idx': target_class_idx,  # 0-19 index for your model
                'class_name': predicted_class_name,  # Without numeric prefix
                'cub_class_id': predicted_class_id,  # Original CUB class ID
                'confidence': confidence,
                'vote_count': f"{vote_count}/{k}",
                'closest_match_id': closest_match['image_id'],
                'closest_match_path': closest_match['image_path'],
                'closest_distance': closest_distance,
                'top_k_classes': ', '.join(top_classes)
            })
        else:
            results.append({
                'test_image_path': str(test_path),
                'test_image_name': test_path.name,
                'class_idx': None,
                'class_name': None,
                'cub_class_id': None,
                'confidence': 0,
                'vote_count': '0/0',
                'closest_match_id': None,
                'closest_match_path': None,
                'closest_distance': None,
                'top_k_classes': None
            })
    
    results_df = pd.DataFrame(results)
    
    # Display statistics
    valid_results = results_df[results_df['class_idx'].notna()]
    
    print(f"\n{'='*70}")
    print("MATCHING RESULTS")
    print(f"{'='*70}")
    print(f"Successfully matched: {len(valid_results)}/{len(results_df)}")
    
    if len(valid_results) > 0:
        print(f"\nConfidence distribution:")
        print(f"  High (≥80%): {len(valid_results[valid_results['confidence'] >= 0.8])}")
        print(f"  Medium (60-80%): {len(valid_results[(valid_results['confidence'] >= 0.6) & (valid_results['confidence'] < 0.8)])}")
        print(f"  Low (<60%): {len(valid_results[valid_results['confidence'] < 0.6])}")
        
        print(f"\nAverage confidence: {valid_results['confidence'].mean()*100:.1f}%")
        print(f"Average distance to closest match: {valid_results['closest_distance'].mean():.4f}")
    
    return results_df

def main():
    # ============= CONFIGURATION =============
    CUB_ROOT = 'CUB_200_2011'
    TEST_DIR = 'Birds_Image_Classification/bdma-07-competition-2026/BDMA7_project_files/test_images/mistery_cat'
    OUTPUT_CSV = 'test_labels_deep_learning.csv'
    K_NEIGHBORS = 5  # Number of neighbors to vote (3, 5, or 10)
    CACHE_FILE = 'cub_features_resnet50_filtered.npz'
    
    # ============= TARGET CLASSES (20 CLASSES IN YOUR TEST SET) =============
    TARGET_CLASSES = [
        'Groove_billed_Ani',
        'Red_winged_Blackbird',
        'Rusty_Blackbird',
        'Gray_Catbird',
        'Brandt_Cormorant',
        'Eastern_Towhee',
        'Indigo_Bunting',
        'Brewer_Blackbird',
        'Painted_Bunting',
        'Bobolink',
        'Lazuli_Bunting',
        'Yellow_headed_Blackbird',
        'American_Crow',
        'Fish_Crow',
        'Brown_Creeper',
        'Yellow_billed_Cuckoo',
        'Yellow_breasted_Chat',
        'Black_billed_Cuckoo',
        'Gray_crowned_Rosy_Finch',
        'Bronzed_Cowbird'
    ]
    # =========================================================================
    
    print("="*70)
    print("Deep Learning Image Matcher (ResNet50)")
    print("="*70)
    print(f"K-nearest neighbors: {K_NEIGHBORS}")
    print(f"Target classes: {len(TARGET_CLASSES)}")
    
    # Validate paths
    if not os.path.exists(CUB_ROOT):
        print(f"\n❌ ERROR: CUB dataset not found: {CUB_ROOT}")
        return
    
    if not os.path.exists(TEST_DIR):
        print(f"\n❌ ERROR: Test directory not found: {TEST_DIR}")
        return
    
    # Initialize feature extractor
    print("\nInitializing ResNet50 feature extractor...")
    extractor = FeatureExtractor()
    
    # Load metadata (filtered to target classes only)
    metadata = load_cub_metadata(CUB_ROOT, filter_classes=TARGET_CLASSES)
    
    if len(metadata) == 0:
        print("\n❌ ERROR: No images found for target classes!")
        print("Check that class names match CUB dataset format")
        return
    
    # Extract/load CUB features
    cub_features, image_ids = extract_cub_features(
        CUB_ROOT, metadata, extractor, CACHE_FILE
    )
    
    # Match test images
    results_df = match_test_images(
        TEST_DIR, cub_features, metadata, image_ids, extractor, K_NEIGHBORS, TARGET_CLASSES
    )
    
    if len(results_df) == 0:
        print("\n❌ No results to save")
        return
    
    # Save results
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Results saved to: {OUTPUT_CSV}")
    
    # Save high confidence only
    high_conf = results_df[results_df['confidence'] >= 0.8]
    if len(high_conf) > 0:
        high_conf_file = OUTPUT_CSV.replace('.csv', '_high_conf.csv')
        high_conf.to_csv(high_conf_file, index=False)
        print(f"✅ High confidence (≥80%): {high_conf_file}")
    
    # Summary
    valid = results_df[results_df['class_idx'].notna()]
    if len(valid) > 0:
        print(f"\n{'='*70}")
        print("DATASET SUMMARY")
        print(f"{'='*70}")
        print(f"Unique classes predicted: {valid['class_idx'].nunique()}")
        print(f"\nPredicted class distribution:")
        class_counts = valid['class_name'].value_counts()
        for i, (class_name, count) in enumerate(class_counts.items(), 1):
            # Get the class_idx for this class
            class_idx = valid[valid['class_name'] == class_name].iloc[0]['class_idx']
            print(f"  {i}. [{class_idx}] {class_name}: {count} images")

if __name__ == "__main__":
    main()