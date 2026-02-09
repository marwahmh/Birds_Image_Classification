#!/bin/bash
#SBATCH --job-name=dino_ssl_birds
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/workdir/cardilloa/Marwa/Birds_Image_Classification/logs/dino_ssl_birds_%j.out
#SBATCH --error=/gpfs/workdir/cardilloa/Marwa/Birds_Image_Classification/logs/dino_ssl_birds_%j.err


# Optional: load your modules / conda env here
# module load python/3.10 cuda/12.1
# source activate your_env
module restore project_env
source activate project_env_twelve


echo "Job started on $(date)"
echo "Running on host: $(hostname)"

cd /gpfs/workdir/cardilloa/Marwa/Birds_Image_Classification
mkdir -p logs


# Install dependencies locally if needed (comment out if using a managed env)
# pip install --user timm torch torchvision pandas scikit-learn tqdm

python DINO_Experiments/dino_ssl_pipeline.py 

echo "Job finished on $(date)" 