#!/bin/bash
#SBATCH --job-name=aasist-train
#SBATCH --account=ug
#SBATCH --partition=partition1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/cluster/workspace/deepvoice/yubeen/aasist/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/cluster/workspace/deepvoice/yubeen/aasist/slurm/logs/%x_%j.err
#SBATCH --chdir=/mnt/cluster/workspace/deepvoice/yubeen/aasist

# Train DeepVoiceAASIST (VF/MF/FF fake detection) on the v3 mixed dataset.
# Mirrors yubeen/ast/slurm/train_ast.sh -- see that file's comment for why
# every absolute path below stays under /mnt/data1/ (compute nodes don't see
# the /data1/... mount the login node uses).
#
#   sbatch train_aasist.sh                              # frozen-backbone baseline
#   sbatch train_aasist.sh --unfreeze-epoch 3           # fine-tune the backbone
#   sbatch train_aasist.sh --limit 2000 --epochs 2      # quick smoke run
#
# Any arguments given to sbatch are forwarded straight to train.py.

set -euo pipefail

PYTHON=/mnt/cluster/workspace/envs/deepvoice/bin/python
AASIST_DIR=/mnt/cluster/workspace/deepvoice/yubeen/aasist

cd "$AASIST_DIR"

export HF_HOME=/mnt/cluster/workspace/.cache/huggingface
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# DataLoader workers each spawn threads; without this they oversubscribe the
# allocated cores and slow the job down.
export OMP_NUM_THREADS=1

echo "node   : $(hostname)"
echo "job    : ${SLURM_JOB_ID}"
echo "gpus   : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "args   : $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "----------------------------------------"

srun --chdir="$AASIST_DIR" "$PYTHON" train.py \
    --epochs 10 \
    --batch-size 16 \
    --num-workers 4 \
    --amp \
    --eval-test \
    --out "checkpoints/job_${SLURM_JOB_ID}" \
    "$@"
