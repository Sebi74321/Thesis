#!/usr/bin/env bash
#SBATCH --job-name=thesis-xai-ablation
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script from the repository root}"
source activate_thesis310.sh
cd CTAB-GAN-Plus-main
python -m xai_reweighting.run_ablation \
  --config configs/mimic_ctabgan.json \
  --stage val \
  --device cuda
