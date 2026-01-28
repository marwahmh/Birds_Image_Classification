# ============================================
# Configuration for Bird Classification
# ============================================
# This file contains all experiment settings.
# Modify values here and they will be automatically loaded in the notebook.

class Config:
    """
    Configuration class for bird image classification experiments.
    
    Sections:
    - Model settings
    - Transfer learning settings
    - Training hyperparameters
    - K-Fold cross validation
    - Image settings
    - Test Time Augmentation (TTA)
    - Voting methods
    - Data augmentation
    - Early stopping
    - Checkpoint settings
    """
    
    # ============================================
    # MODEL SETTINGS
    # ============================================
    # Available EfficientNetV2 models (uncomment one):
    model_name = 'tf_efficientnetv2_s.in21k_ft_in1k'  # EfficientNetV2-S pretrained on ImageNet
    # model_name = 'tf_efficientnetv2_s.in21k'
    # model_name = 'tf_efficientnetv2_m.in21k_ft_in1k'
    # model_name = 'tf_efficientnetv2_m.in21k'
    # model_name = 'tf_efficientnetv2_l.in21k_ft_in1k'
    # model_name = 'tf_efficientnetv2_l.in21k'

    num_classes = 20
    
    # ============================================
    # CLASSIFIER HEAD SETTINGS
    # ============================================
    # Head type options:
    #   'default'  - Single linear layer (timm default): Linear(1280 → num_classes)
    #   'mlp'      - MLP with hidden layer: Linear → BN → ReLU → Dropout → Linear
    #   'mlp_deep' - Deeper MLP: Linear → BN → ReLU → Dropout → Linear → BN → ReLU → Dropout → Linear
    head_type = 'mlp'
    
    # Hidden layer dimensions (only used if head_type is 'mlp' or 'mlp_deep')
    head_hidden_dim = 512       # First hidden layer size
    head_hidden_dim_2 = 256     # Second hidden layer size (only for 'mlp_deep')
    
    # Head dropout rate (applied in custom heads)
    head_dropout = 0.3
    
    # Use batch normalization in custom heads
    head_use_batchnorm = True
    
    # ============================================
    # TRANSFER LEARNING SETTINGS
    # ============================================
    use_transfer_learning = True  # Enable transfer learning strategy
    freeze_backbone_epochs = 10   # Number of epochs with frozen backbone (train only head)
    unfreeze_epochs = 20           # Number of epochs after unfreezing (fine-tune layers)
    # Total epochs = freeze_backbone_epochs + unfreeze_epochs
    
    # Freeze Backbone Always
    # If True, keep backbone frozen throughout entire training (only train classifier)
    # Stage 2 will be skipped and only classifier head is trained for all epochs
    freeze_backbone_always = False
    
    # Progressive Unfreezing Strategy (only used if freeze_backbone_always=False)
    unfreeze_all_layers = False   # If True, unfreeze all layers. If False, use progressive unfreezing
    unfreeze_last_n_blocks = 2    # Number of last blocks/stages to unfreeze (only if unfreeze_all_layers=False)
    # For EfficientNetV2-S: typically has 7 blocks, unfreezing last 2-3 blocks is often optimal
    
    # ============================================
    # TRAINING HYPERPARAMETERS
    # ============================================
    batch_size = 16
    learning_rate_head = 1e-3     # Higher LR for classifier head (when backbone frozen)
    learning_rate_backbone = 1e-5 # Lower LR for backbone (when unfrozen)
    weight_decay = 1e-4           # Helps prevent overfitting by discouraging large weights
    
    # ============================================
    # K-FOLD CROSS VALIDATION
    # ============================================
    n_folds = 3
    
    # ============================================
    # IMAGE SETTINGS
    # ============================================
    img_size = 384  # EfficientNetV2-S optimal size
    # More experimenting to find the best size is needed (maybe cropping using Deep learning models to fixed size)
    
    # ============================================
    # TEST TIME AUGMENTATION (TTA) SETTINGS
    # ============================================
    tta_transforms = 5  # Number of TTA augmentations
    
    # ============================================
    # VOTING METHOD SETTINGS
    # ============================================
    # Options: 'mean' (soft voting - average probabilities) or 'majority' (hard voting - count votes)
    tta_voting = 'mean'        # Voting method for TTA predictions: 'mean' or 'majority'
    ensemble_voting = 'mean'   # Voting method for model ensemble: 'mean' or 'majority'
    
    # ============================================
    # DATA AUGMENTATION SETTINGS
    # ============================================
    rotation_degrees = 15
    color_jitter = {
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1
    }
    random_erasing_prob = 0.25
    
    # ============================================
    # EARLY STOPPING
    # ============================================
    patience = 10  # Number of epochs without improvement before stopping
    
    # ============================================
    # CHECKPOINT SETTINGS
    # ============================================
    # Checkpoint Selection Strategy:
    # 'best': Use the model with the best validation accuracy (default)
    # 'last': Use the model from the last epoch
    checkpoint_strategy = 'best'
    
    # Save all epoch checkpoints for later manual selection
    # If True, saves checkpoint after each epoch (uses more disk space)
    # If False, only saves final selected model per fold
    save_all_epochs = True
    
    # Manual fold selection for ensemble (None = use all folds)
    # Set to a list of fold numbers to use specific folds, e.g., [1, 3, 4]
    # Useful after reviewing individual fold performance
    selected_folds = None  # e.g., [1, 2, 3, 4, 5] or [1, 3, 5] for specific folds
    
    # ============================================
    # COMPUTED PROPERTIES
    # ============================================
    @property
    def total_epochs(self):
        """Total training epochs"""
        if not self.use_transfer_learning:
            return 15
        elif self.freeze_backbone_always:
            # If always frozen, use freeze_backbone_epochs + unfreeze_epochs (but all with frozen backbone)
            return self.freeze_backbone_epochs + self.unfreeze_epochs
        else:
            # Normal two-stage: freeze epochs + unfreeze epochs
            return self.freeze_backbone_epochs + self.unfreeze_epochs


# ============================================
# QUICK CONFIGURATION PRESETS
# ============================================
# Uncomment a preset function call at the bottom to use predefined settings

def preset_quick_test(config):
    """Quick test settings for debugging"""
    config.freeze_backbone_epochs = 2
    config.unfreeze_epochs = 0
    config.n_folds = 2
    config.batch_size = 8
    config.save_all_epochs = False
    return config

def preset_frozen_backbone(config):
    """Train only classifier head (fast, memory efficient)"""
    config.freeze_backbone_always = True
    config.freeze_backbone_epochs = 15
    config.unfreeze_epochs = 0
    return config

def preset_two_stage_progressive(config):
    """Two-stage training with progressive unfreezing"""
    config.freeze_backbone_always = False
    config.freeze_backbone_epochs = 10
    config.unfreeze_epochs = 5
    config.unfreeze_all_layers = False
    config.unfreeze_last_n_blocks = 2
    return config

def preset_two_stage_full(config):
    """Two-stage training with full fine-tuning"""
    config.freeze_backbone_always = False
    config.freeze_backbone_epochs = 10
    config.unfreeze_epochs = 5
    config.unfreeze_all_layers = True
    return config


# ============================================
# USAGE
# ============================================
# In the notebook, import and use:
#
#   from config import Config, preset_quick_test
#   config = Config()
#   
#   # Optionally apply a preset:
#   # config = preset_quick_test(config)
#
#   # Or modify individual settings:
#   # config.batch_size = 32
#   # config.n_folds = 3
