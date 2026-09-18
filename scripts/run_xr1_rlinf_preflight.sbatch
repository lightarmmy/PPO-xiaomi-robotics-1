#!/usr/bin/env bash
#SBATCH --job-name=xr1-ppo-preflight
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL/Xiaomi-Robotics-1/logs/xr1_rlinf_preflight_%j.log

set -euo pipefail
# Slurm executes a staged copy of this script under /var/spool; use the
# directory from which sbatch was submitted so paths resolve on compute nodes.
repo_root=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
repo_root=$(cd "${repo_root}" && pwd)
export PYTHONPATH=${repo_root}/xr1:${repo_root}/third_party/RLinf:${repo_root}:${PYTHONPATH:-}
export XR1_ROBOCASA365_CKPT=${XR1_ROBOCASA365_CKPT:-${repo_root}/checkpoints/Xiaomi-Robotics-1-RoboCasa365}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export TOKENIZERS_PARALLELISM=false
cd "${repo_root}"
exec "${repo_root}/.conda-robocasa365/bin/python" scripts/xr1_rlinf_preflight.py
