#!/bin/bash
#SBATCH --job-name=deepvoice_web_demo
#SBATCH --account=ug
#SBATCH --partition=partition1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --chdir=/mnt/data1/dsl07/deepvoice/sewon/web_demo/backend
#SBATCH --output=/mnt/data1/dsl07/deepvoice/sewon/web_demo/slurm/logs/%x_%j.out
#SBATCH --error=/mnt/data1/dsl07/deepvoice/sewon/web_demo/slurm/logs/%x_%j.err

export PATH="/mnt/data1/dsl07/envs/deepvoice/bin:$PATH"

# onnxruntime-gpu's CUDA provider .so does a plain dlopen() for libcudnn/
# libnvrtc/etc and does not know about pip-installed nvidia-*-cu12 package
# locations the way torch's own loader does. Without this, BS PolarFormer
# silently falls back to CPU (slower, and CPU-side memory use can spike).
export LD_LIBRARY_PATH="$(python -c "
import glob, os, nvidia
site_nvidia = os.path.dirname(nvidia.__file__)
dirs = sorted(glob.glob(os.path.join(site_nvidia, '*', 'lib')))
print(':'.join(dirs))
"):${LD_LIBRARY_PATH:-}"

bash run_on_slurm.sh
