# Birds Image Classification

Project repository for bird species image classification experiments used in BDMA7 competition and research. This workspace contains datasets, training and evaluation scripts, experiment notebooks, and model checkpoints for CNN, DINO (self-supervised), and Transformer-based approaches.

## Key Contents

- `bdma-07-competition-2026/`: competition data and sample submissions.
- `CUB_200_2011/`: CUB-200-2011 dataset local copy (metadata and images).
- `CNN_Experiments/`: CNN-based experiments, configs, models, and per-class accuracy scripts.
- `DINO_Experiments/`: Self-supervised DINO/DINOv2 experiments and submission scripts.
- `Transformer_Experiments/`: Transformer & SWIN/VIT experiments.
- Top-level scripts and notebooks: `classify_test_with_clip.py`, `find_test_labels.py`, `find_test_labels_embeddings.py`, `Data_Exploration.ipynb`.

## Overview

This repository explores multiple approaches for fine-grained bird species classification, including:

- Supervised CNN training and evaluation (EfficientNet, ConvNeXt, etc.).
- Self-supervised pretraining with DINO/DINOv2, followed by fine-tuning.
- Transformer-based models and hybrid pipelines.
- Embedding-based label-matching and CLIP-assisted inference for test-time label matching.

Use cases:

- Reproduce experiments and per-class accuracy analysis.
- Generate submission CSVs for the BDMA competition.
- Inspect model misclassifications and debug class-wise performance.

## Requirements

- Python 3.9+ recommended
- Common packages: `torch`, `torchvision`, `timm`, `numpy`, `pandas`, `scikit-learn`, `Pillow`, `opencv-python`, `matplotlib`, `tqdm`.

If a `requirements.txt` is not present, create a virtualenv and install the packages you need. Example:

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision timm numpy pandas scikit-learn Pillow opencv-python matplotlib tqdm
```

## Data layout

- Place the BDMA competition data under `bdma-07-competition-2026/` (already included here in this workspace snapshot).
- The CUB dataset is under `CUB_200_2011/` with `images/`, `train_test_split.txt`, and label files.

Before running training or evaluation, ensure dataset paths in the config files or scripts point to the correct folders.

## Common Scripts & How to Run

- Training / fine-tuning (example): update the config in `CNN_Experiments/config.py` or your training script, then run the training entrypoint used by the experiment. Example pattern:

```bash
python train.py --config CNN_Experiments/config.py --exp-name convnexttiny
```

- Evaluate a model and compute per-class accuracy:

```bash
python CNN_Experiments/compute_per_class_accuracy.py --predictions submissions/convnexttiny/pred.csv --labels bdma-07-competition-2026/val_metadata.csv
```

- Run CLIP-based test classification (example):

```bash
python classify_test_with_clip.py --images bdma-07-competition-2026/test_images --labels-file bdma-07-competition-2026/BDMA7_project_files/test_labels.txt --output submission_clip.csv
```

- Find and match test labels via embeddings:

```bash
python find_test_labels_embeddings.py --embeddings-dir embeddings/ --output test_labels_embeddings.csv
```

Adjust arguments to match your paths and preferred checkpoints.

## Notebooks

- `Data_Exploration.ipynb`: exploratory data analysis and visualization of the datasets.
- `CNN_Experiments/CNN_Experiments.ipynb`: experiments and interactive training/evaluation snippets.

Open these in Jupyter or VS Code for interactive runs and plots.

## Experiments and Results

- Experiment outputs and saved models are stored under each experiments folder (e.g., `CNN_Experiments/models/`, `DINO_Experiments/results/`).
- Submissions are under `submissions/` with subfolders per model.
- Per-class accuracy JSON files and other metrics are stored alongside experiment scripts (see `per_class_accuracy_results.json`).

## Submissions

When producing a submission CSV for the competition, follow the sample format in `bdma-07-competition-2026/sample_submission.csv`. Use the provided scripts to export model predictions into the required CSV format.

## Contributing

- Keep experiments reproducible: log the command-line args, dataset splits, and checkpoint paths.
- If adding new models or experiments, add a small README inside the experiment folder describing how to run it.

## Troubleshooting & Tips

- GPU: ensure CUDA and the matching PyTorch are installed when training on GPU.
- Large datasets: point scripts to dataset directories rather than copying data around.
- Checkpoint compatibility: note the model architecture and library versions used to save checkpoints.

## Contact

For questions or collaboration, open an issue in this repository or contact the project maintainer.

---

If you'd like, I can also:

- generate a `requirements.txt` from the imports found in the repo,
- add short README files for each experiments folder,
- or create runnable example commands for a specific model (tell me which one).
