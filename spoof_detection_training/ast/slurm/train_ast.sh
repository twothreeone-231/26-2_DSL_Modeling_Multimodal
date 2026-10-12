#!/bin/bash
#SBATCH --job-name=ast-train
#SBATCH --account=ug
#SBATCH --partition=partition1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/cluster/workspace/deepvoice/yubeen/ast/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/cluster/workspace/deepvoice/yubeen/ast/slurm/logs/%x_%j.err
#SBATCH --chdir=/mnt/cluster/workspace/deepvoice/yubeen/ast

# Train DeepVoiceAST on the v3 mixed dataset.
#
#   sbatch train_ast.sh                              # frozen-backbone baseline
#   sbatch train_ast.sh --unfreeze-epoch 3           # fine-tune the backbone
#   sbatch train_ast.sh --limit 2000 --epochs 2      # quick smoke run
#
# Any arguments given to sbatch are forwarded straight to train.py.
#
# [path note] the login node (hpcmaster) sees this shared storage at /data1/...,
# but the compute nodes this job actually lands on (e.g. hpc-stat1) only see it
# at /mnt/data1/... (confirmed by hand: pointing --output at /data1/... makes
# slurmd fail to even open the log file, so the job dies instantly with
# ExitCode 0:53 and no output at all -- this is what was happening before).
# Every absolute path below must stay under /mnt/data1/.

set -euo pipefail

PYTHON=/mnt/cluster/workspace/envs/deepvoice/bin/python
AST_DIR=/mnt/cluster/workspace/deepvoice/yubeen/ast

cd "$AST_DIR"

# Notifications are handled from an external watcher script to avoid
# Slurm job termination when mail is invoked inside the job itself.

# Keep HF weights on the shared volume so every node reuses one download
# instead of re-fetching into a home dir that may not be shared.
export HF_HOME=/mnt/cluster/workspace/.cache/huggingface
# The AST checkpoint is already cached above; fail fast instead of hanging
# if a compute node has no outbound internet access.
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

srun --chdir="$AST_DIR" "$PYTHON" train.py \
    --epochs 10 \
    --batch-size 16 \
    --num-workers 4 \
    --amp \
    --eval-test \
    --out "checkpoints/job_${SLURM_JOB_ID}" \
    "$@"
