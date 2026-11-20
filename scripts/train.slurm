#!/bin/bash
#SBATCH --job-name=date-grpo
#SBATCH --output=logs/grpo-%j.out
#SBATCH --error=logs/grpo-%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
# Set Scholar-specific account, partition, and GPU options according to RCAC documentation.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
python scripts/train.py --data data/dates.csv --output outputs/date-grpo --steps 100
