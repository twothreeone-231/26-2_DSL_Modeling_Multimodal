#!/bin/bash
#SBATCH --job-name=presence-train
#SBATCH --account=ug
#SBATCH --partition=partition1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/data1/dsl07/deepvoice/yubeen/ast/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/data1/dsl07/deepvoice/yubeen/ast/slurm/logs/%x_%j.err
#SBATCH --chdir=/mnt/data1/dsl07/deepvoice/yubeen/ast

# Train the VP/MP presence head on the v3 mixed dataset.
#
#   sbatch train_presence.sh                              # frozen-backbone baseline
#   sbatch train_presence.sh --unfreeze-epoch 3           # fine-tune the backbone
#   sbatch train_presence.sh --limit 2000 --epochs 2      # quick smoke run
#
# Any arguments given to sbatch are forwarded straight to presence.py.
#
# [path note] mirrors train_ast.sh: the login node sees /data1/..., but
# compute nodes (e.g. hpc-stat1) only see /mnt/data1/... - keep every
# absolute path below under /mnt/data1/.

set -euo pipefail

PYTHON=/mnt/data1/dsl07/envs/deepvoice/bin/python
AST_DIR=/mnt/data1/dsl07/deepvoice/yubeen/ast

cd "$AST_DIR"

export HF_HOME=/mnt/data1/dsl07/.cache/huggingface
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

echo "node   : $(hostname)"
echo "job    : ${SLURM_JOB_ID}"
echo "gpus   : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "args   : $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "----------------------------------------"

srun --chdir="$AST_DIR" "$PYTHON" presence.py \
    --epochs 10 \
    --batch-size 16 \
    --num-workers 4 \
    --amp \
    --eval-test \
    --out "checkpoints/job_${SLURM_JOB_ID}" \
    "$@"
