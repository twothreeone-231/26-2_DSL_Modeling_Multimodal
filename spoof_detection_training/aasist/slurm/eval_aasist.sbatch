#!/bin/bash
#
# Evaluate a trained DeepVoiceAASIST checkpoint on val + test splits.
# 사용법: sbatch slurm/eval_aasist.sbatch --checkpoint checkpoints/job_<id>/aasist_best.pt [evaluation.py에 그대로 전달할 인자...]
#
#SBATCH --job-name=aasist-eval
#SBATCH --account=ug
#SBATCH --partition=partition1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --chdir=/mnt/cluster/workspace/deepvoice/yubeen/aasist
#SBATCH --output=/mnt/cluster/workspace/deepvoice/yubeen/aasist/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/cluster/workspace/deepvoice/yubeen/aasist/slurm/logs/%x_%j.err

set -euo pipefail

PYTHON=/mnt/cluster/workspace/envs/deepvoice/bin/python
AASIST_DIR=/mnt/cluster/workspace/deepvoice/yubeen/aasist

export HF_HOME=/mnt/cluster/workspace/.cache/huggingface
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

echo "node: $(hostname)"
echo "job : ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$PYTHON" evaluation.py \
    --split both \
    --batch-size 16 \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    "$@"

echo "===== DONE ====="
