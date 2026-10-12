#!/bin/bash
#
# Evaluate a trained DeepVoiceAST checkpoint on val + test splits.
# 사용법: sbatch slurm/eval_ast.sbatch [evaluation.py에 그대로 전달할 인자...]
#   기본값: checkpoints/job_228239/ast_best.pt (현재 프로젝트에서 유일하게 존재하는 체크포인트)
#
#SBATCH --job-name=ast-eval
#SBATCH --account=ug
#SBATCH --partition=partition1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --chdir=/mnt/data1/dsl07/deepvoice/yubeen/ast
#SBATCH --output=/mnt/data1/dsl07/deepvoice/yubeen/ast/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/data1/dsl07/deepvoice/yubeen/ast/slurm/logs/%x_%j.err

set -euo pipefail

PYTHON=/mnt/data1/dsl07/envs/deepvoice/bin/python
AST_DIR=/mnt/data1/dsl07/deepvoice/yubeen/ast

export HF_HOME=/mnt/data1/dsl07/.cache/huggingface
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

echo "node: $(hostname)"
echo "job : ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$PYTHON" evaluation.py \
    --checkpoint "$AST_DIR/checkpoints/job_228239/ast_best.pt" \
    --split both \
    --batch-size 16 \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    "$@"

echo "===== DONE ====="
