#!/usr/bin/env bash
#SBATCH --job-name=xr1-direct-rollout
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL/Xiaomi-Robotics-1/logs/xr1_direct_rollout_%j.log
set -euo pipefail
repo_root=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "${repo_root}"
export PYTHONPATH="${repo_root}/xr1:${repo_root}:${PYTHONPATH:-}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export PYTHONFAULTHANDLER=1
exec .conda-robocasa365/bin/python -u scripts/xr1_direct_robocasa365_rollout.py \
  --model-path "${XR1_ROBOCASA365_CKPT:-${repo_root}/checkpoints/Xiaomi-Robotics-1-RoboCasa365}" \
  --task "${XR1_DIRECT_TASK:-TurnOnMicrowave}" \
  --split "${XR1_DIRECT_SPLIT:-pretrain}" \
  --seed "${XR1_DIRECT_SEED:-87}" \
  --output "${XR1_DIRECT_OUTPUT:-${repo_root}/results/xr1_direct_rollout_${SLURM_JOB_ID}.json}"
